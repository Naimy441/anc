"""Noise-suppression engines.

Each engine consumes mono float32 audio at 48 kHz in blocks that are a multiple of
``engine.hop`` and returns the same number of samples, delayed by exactly
``engine.latency`` samples. The fixed, known latency lets the pipeline time-align
the dry signal for click-free bypass and precise wet/dry ("reduction %") mixing.

Engines
-------
* ``neural``   – DeepFilterNet3 (Schröter et al., 2023), a full-band 48 kHz
                 deep-filtering network, run as a fully streaming ONNX graph.
* ``spectral`` – Classic statistical enhancer: SPP-based MMSE noise PSD tracking
                 (Gerkmann & Hendriks 2012) + decision-directed a-priori SNR +
                 OM-LSA gain (Cohen 2002). Very low latency and CPU.
* ``hybrid``   – neural followed by a gentle spectral residual-noise suppressor.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.special import exp1

SAMPLE_RATE = 48000
ADAPTIVE_BIAS = 1.25
MS_BIAS = 2.2
MODEL_DIR = Path(__file__).resolve().parent / "models" / "deepfilternet3"


class Engine:
    name = "base"
    hop = 512
    latency = 0

    def process(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def configure(self, settings: dict) -> None:
        pass

    # Optional extras surfaced in telemetry.
    def speech_prob(self) -> float | None:
        return None


class PassthroughEngine(Engine):
    name = "off"
    hop = 128
    latency = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        return x


# ---------------------------------------------------------------------------
# DeepFilterNet3 (streaming ONNX)
# ---------------------------------------------------------------------------


class DeepFilterNetEngine(Engine):
    """DeepFilterNet3 with explicit recurrent / ring-buffer state per 512-sample hop.

    The ONNX graph comes from the torchDF project (pure-torch streaming port of
    DeepFilterNet3, fine-tuned for hop 512 / FFT 1024). Algorithmic latency is
    FFT-hop (512) + 2 frames of lookahead (1024) = 1536 samples (32 ms).
    """

    name = "neural"
    hop = 512
    latency = 1536

    def __init__(self, model_path: Path | None = None, threads: int = 1):
        import onnxruntime as ort

        path = Path(model_path or MODEL_DIR / "denoiser_model.onnx")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        # Keep worker threads hot: avoids wake-up jitter on every 10 ms frame.
        so.add_session_config_entry("session.intra_op.allow_spinning", "1")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.state_inputs = [i for i in self.sess.get_inputs() if i.name != "input_frame"]
        self.state_names = [i.name for i in self.state_inputs]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.reset()

    def reset(self) -> None:
        self.state = {}
        for i in self.state_inputs:
            self.state[i.name] = np.zeros(i.shape, dtype=np.float32)
        self.state["erb_norm_state"] = np.linspace(-60.0, -90.0, 32, dtype=np.float32)
        self.state["band_unit_norm_state"] = np.linspace(
            0.001, 0.0001, 96, dtype=np.float32
        ).reshape(1, 96, 1)
        self._sp = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        feeds = self.state
        for k in range(0, len(x), self.hop):
            frame = np.ascontiguousarray(x[k : k + self.hop], dtype=np.float32)
            feeds["input_frame"] = frame
            res = self.sess.run(self.out_names, feeds)
            out[k : k + self.hop] = res[0]
            for n, v in zip(self.state_names, res[1:]):
                feeds[n] = v
        feeds.pop("input_frame", None)
        return out


# ---------------------------------------------------------------------------
# Statistical spectral enhancer
# ---------------------------------------------------------------------------


def _conv_bins(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Smooth each row along frequency with kernel ``k`` (edge-normalised)."""
    pad = len(k) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(x)
    for i, w in enumerate(k):
        out += w * xp[:, i : i + x.shape[1]]
    return out


def _soft_threshold(z: np.ndarray, lo_db: float = -10.0, hi_db: float = -5.0) -> np.ndarray:
    """Cohen's log-domain ramp: 0 below lo, 1 above hi."""
    zdb = 10.0 * np.log10(z + 1e-12)
    return np.clip((zdb - lo_db) / (hi_db - lo_db), 0.0, 1.0)


class SpectralEngine(Engine):
    """OM-LSA enhancer with SPP-based noise tracking and an optional learned profile.

    STFT: 1024-point sqrt-Hann WOLA, 75 % overlap (hop 256) -> latency 768 samples.

    The engine can evaluate several (over-subtraction, gain-floor) configurations
    side by side on the same input (``set_candidates``); the auto-tuner uses this to
    score dozens of settings in one pass with exactly the runtime code.
    """

    name = "spectral"

    def __init__(self, fft: int = 1024, hop: int = 256, residual: bool = False):
        self.fft = fft
        self.hop = hop
        self.latency = fft - hop
        self.bins = fft // 2 + 1
        n = np.arange(fft)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / fft)  # periodic Hann
        self.win = np.sqrt(hann).astype(np.float32)
        # WOLA normalisation so analysis*synthesis windows overlap-add to 1.
        self.norm = np.float32(hop / np.sum(hann))
        self.residual = residual
        self.noise_mode = "adaptive"  # adaptive | learned | hybrid
        self.learned: np.ndarray | None = None
        self.track_learned = True
        self.dd_alpha = 0.98
        self._band = slice(int(300 * fft / SAMPLE_RATE), int(4000 * fft / SAMPLE_RATE))
        self._init_frames = max(4, int(0.06 * SAMPLE_RATE / hop))
        self._ms_sub = max(1, int(0.19 * SAMPLE_RATE / hop))  # 8 sub-windows ≈ 1.5 s
        self._slow_alpha = 1.0 - 0.97 ** ((hop / SAMPLE_RATE) / 0.00533)
        self._track_alpha = 0.8 ** ((hop / SAMPLE_RATE) / 0.016)
        self._k_local = np.hanning(5)[1:-1] / np.hanning(5)[1:-1].sum()
        self._k_global = np.hanning(33)[1:-1] / np.hanning(33)[1:-1].sum()
        self._set_params([1.0], [-18.0 if residual else -40.0])
        self._batch = False

    def _set_params(self, betas, floors_db) -> None:
        self._beta = np.asarray(betas, dtype=np.float64).reshape(-1, 1)
        self._gmin = (10.0 ** (np.asarray(floors_db, dtype=np.float64) / 20.0)).reshape(-1, 1)
        self.reset()

    def set_candidates(self, betas, floors_db) -> None:
        """Run len(betas) configurations in parallel; ``process`` then returns (C, n)."""
        self._batch = True
        self._set_params(betas, floors_db)

    def reset(self) -> None:
        c = len(self._beta)
        self.inbuf = np.zeros(self.fft, dtype=np.float32)
        self.outbuf = np.zeros((c, self.fft), dtype=np.float32)
        self.noise = None
        self.noise_slow = None
        self.ms_smooth = np.zeros(self.bins)
        self.ms_cur = np.full(self.bins, np.inf)
        self.ms_ring = np.full((8, self.bins), np.inf)
        self.ms_pos = 0
        self.ms_count = 0
        self.prev_clean_pow = np.zeros((c, self.bins))
        self.prev_xi = np.zeros((c, self.bins))
        self.zeta = np.zeros((c, self.bins))
        self._gain_prev = np.broadcast_to(self._gmin, (c, self.bins)).copy()
        self.spp_smooth = np.zeros(self.bins)
        self.frames = 0
        self._sp = 0.0

    def configure(self, s: dict) -> None:
        self.noise_mode = s.get("noise_mode", "adaptive")
        smooth = float(s.get("spectral_smoothing", 0.98))
        self.dd_alpha = min(0.995, max(0.8, smooth))
        if self._batch:
            return
        if self.residual:
            # Behind the neural net we only remove what is left.
            beta = float(s.get("post_aggressiveness", 1.0))
            floor = float(s.get("post_floor_db", -18.0))
        else:
            beta = float(s.get("spectral_aggressiveness", 1.0))
            floor = float(s.get("spectral_floor_db", -40.0))
        if beta != float(self._beta[0, 0]) or abs(10 ** (floor / 20) - float(self._gmin[0, 0])) > 1e-9:
            self._beta[0, 0] = beta
            self._gmin[0, 0] = 10 ** (floor / 20)
            if self.frames == 0:
                self._gain_prev[:] = self._gmin

    def psd_of(self, audio: np.ndarray) -> np.ndarray:
        """Mean power spectrum of ``audio`` measured with this engine's analysis window."""
        h, f = self.hop, self.fft
        n = (len(audio) - f) // h + 1
        if n <= 0:
            raise ValueError("audio too short for a noise profile")
        idx = np.arange(f)[None, :] + h * np.arange(n)[:, None]
        spec = np.fft.rfft(audio[idx] * self.win, axis=1)
        return np.mean(spec.real ** 2 + spec.imag ** 2, axis=0) + 1e-12

    def speech_prob(self) -> float | None:
        return self._sp

    # -- core --------------------------------------------------------------------
    def _frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.win)
        pw = spec.real * spec.real + spec.imag * spec.imag + 1e-12

        # --- noise PSD: unbiased MMSE estimator with speech-presence probability.
        # Seed it with the average of the first ~60 ms (a single frame's periodogram
        # leaves bins far too low and the tracker then mistakes them for speech).
        if self.frames < self._init_frames:
            self.noise = pw.copy() if self.noise is None else self.noise + (pw - self.noise) / (self.frames + 1)
            if self.frames == 0:
                self.ms_smooth = pw.copy()
        xi_h1 = 31.62  # 15 dB fixed a-priori SNR under speech presence
        g = xi_h1 / (1.0 + xi_h1)
        post = pw / self.noise
        p = 1.0 / (1.0 + (1.0 + xi_h1) * np.exp(-np.minimum(post * g, 50.0)))
        self.spp_smooth = 0.965 * self.spp_smooth + 0.035 * p
        p = np.where(self.spp_smooth > 0.99, np.minimum(p, 0.99), p)
        est = (1.0 - p) * pw + p * self.noise
        # Gerkmann & Hendriks use 0.8 at a 16 ms hop; rescaled to this hop (~5 ms).
        if self.frames >= self._init_frames:
            self.noise = self._track_alpha * self.noise + (1.0 - self._track_alpha) * est
        self.frames += 1
        noise_only = float(np.mean(p[self._band])) < 0.1

        # Gain uses a slower, bias-compensated copy of the tracker (cf. IMCRA's 1.47
        # factor) – a jittery noise estimate is the main source of musical noise.
        if self.noise_slow is None:
            self.noise_slow = self.noise.copy()
        else:
            self.noise_slow += self._slow_alpha * (self.noise - self.noise_slow)
        # Minimum statistics (Martin 2001) backstop: anything that never drops below
        # its level for ~1.5 s (hum, fan tones) is noise, even if the SPP tracker
        # mistook it for speech and froze.
        self.ms_smooth += 0.1 * (pw - self.ms_smooth)
        np.minimum(self.ms_cur, self.ms_smooth, out=self.ms_cur)
        self.ms_count += 1
        if self.ms_count >= self._ms_sub:
            self.ms_ring[self.ms_pos] = self.ms_cur
            self.ms_pos = (self.ms_pos + 1) % len(self.ms_ring)
            self.ms_cur = self.ms_smooth.copy()
            self.ms_count = 0
        s_min = np.minimum(self.ms_ring.min(axis=0), self.ms_cur)
        lam = np.maximum(np.maximum(self.noise_slow, self.noise), s_min * MS_BIAS) * ADAPTIVE_BIAS
        learned = self.learned
        if learned is not None and self.noise_mode != "adaptive":
            if self.track_learned and noise_only:
                # Follow slow drift of a learned profile during speech pauses (~3 s).
                learned *= 0.998
                learned += 0.002 * pw
            lam = learned if self.noise_mode == "learned" else np.maximum(lam, learned)
        lam = lam * self._beta  # (C, bins)

        # --- decision-directed a-priori SNR + MMSE log-spectral amplitude gain
        gamma = np.minimum(pw / lam, 1e4)
        xi = self.dd_alpha * self.prev_clean_pow / lam + (1 - self.dd_alpha) * np.maximum(
            gamma - 1.0, 0.0
        )
        xi = np.maximum(xi, 10 ** (-25 / 10))
        v = np.maximum(xi * gamma / (1.0 + xi), 1e-8)
        g_lsa = np.minimum(xi / (1.0 + xi) * np.exp(0.5 * exp1(v)), 1.0)

        # --- OM-LSA (Cohen 2001): a-priori speech-absence probability q from the
        # recursively averaged a-priori SNR, smoothed locally (3 bins) and globally
        # (31 bins); then the conditional speech-presence probability p.
        self.zeta = 0.7 * self.zeta + 0.3 * self.prev_xi
        z_loc = _conv_bins(self.zeta, self._k_local)
        z_glob = _conv_bins(self.zeta, self._k_global)
        p_loc = _soft_threshold(z_loc)
        p_glob = _soft_threshold(z_glob)
        z_frame = np.mean(self.zeta[:, self._band], axis=1, keepdims=True)
        p_frame = _soft_threshold(z_frame)
        q = np.minimum(1.0 - p_loc * p_glob * p_frame, 0.95)
        p = 1.0 / (1.0 + q / (1.0 - q) * (1.0 + xi) * np.exp(-np.minimum(v, 50.0)))
        self.prev_xi = xi

        gain = g_lsa ** p * self._gmin ** (1.0 - p)
        # Speech-absent regions: smooth the gain over time so the residual stays a
        # steady, quieter copy of the noise instead of chirping (musical noise).
        sm = self._gain_prev * 0.7 + gain * 0.3
        gain = p * gain + (1.0 - p) * np.minimum(sm, gain * 2.0)
        self._gain_prev = gain
        self.prev_clean_pow = (g_lsa * g_lsa) * pw
        self._sp = float(np.mean(p[0, self._band]))

        out = np.fft.irfft(spec * gain, n=self.fft, axis=-1).astype(np.float32)
        return out * self.win

    def process(self, x: np.ndarray) -> np.ndarray:
        c = len(self._beta)
        out = np.empty((c, len(x)), dtype=np.float32)
        h = self.hop
        for k in range(0, len(x), h):
            self.inbuf[:-h] = self.inbuf[h:]
            self.inbuf[-h:] = x[k : k + h]
            y = self._frame(self.inbuf)
            self.outbuf += y * self.norm
            out[:, k : k + h] = self.outbuf[:, :h]
            self.outbuf[:, :-h] = self.outbuf[:, h:]
            self.outbuf[:, -h:] = 0.0
        return out if self._batch else out[0]


class HybridEngine(Engine):
    """DeepFilterNet3 followed by a residual OM-LSA post-filter."""

    name = "hybrid"

    def __init__(self):
        self.neural = DeepFilterNetEngine()
        self.post = SpectralEngine(residual=True)
        self.hop = self.neural.hop
        self.latency = self.neural.latency + self.post.latency

    def reset(self) -> None:
        self.neural.reset()
        self.post.reset()

    def configure(self, s: dict) -> None:
        self.post.configure(s)

    def process(self, x: np.ndarray) -> np.ndarray:
        return self.post.process(self.neural.process(x))

    def speech_prob(self) -> float | None:
        return self.post.speech_prob()


ENGINES = {
    "neural": "DeepFilterNet3 · neural (48 kHz full-band)",
    "hybrid": "DeepFilterNet3 + OM-LSA residual suppressor",
    "spectral": "OM-LSA statistical (ultra-low latency)",
}


def create_engine(name: str) -> Engine:
    if name == "neural":
        return DeepFilterNetEngine(threads=int(os.environ.get("ANC_ORT_THREADS", "1")))
    if name == "hybrid":
        return HybridEngine()
    if name == "spectral":
        return SpectralEngine()
    if name == "off":
        return PassthroughEngine()
    raise ValueError(f"unknown engine {name!r}")

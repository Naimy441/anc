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


class SpectralEngine(Engine):
    """OM-LSA enhancer with SPP-based noise tracking and optional learned profile.

    STFT: 1024-point sqrt-Hann WOLA, 75 % overlap (hop 256) -> latency 768 samples.
    """

    name = "spectral"

    def __init__(self, fft: int = 1024, hop: int = 256, floor_db: float = -40.0,
                 residual: bool = False):
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
        self.floor_db = floor_db
        self.aggressiveness = 1.0
        self.noise_mode = "adaptive"  # adaptive | learned | hybrid
        self.learned: np.ndarray | None = None
        self._learning = 0
        self._learn_acc = None
        self._learn_n = 0
        self.dd_alpha = 0.98
        self.reset()

    def reset(self) -> None:
        self.inbuf = np.zeros(self.fft, dtype=np.float32)
        self.outbuf = np.zeros(self.fft, dtype=np.float32)
        self.noise = None
        self.prev_clean_pow = np.zeros(self.bins)
        self.spp_smooth = np.zeros(self.bins)
        self.frames = 0
        self._sp = 0.0

    def configure(self, s: dict) -> None:
        self.aggressiveness = float(s.get("spectral_aggressiveness", 1.0))
        self.noise_mode = s.get("noise_mode", "adaptive")
        smooth = float(s.get("spectral_smoothing", 0.98))
        self.dd_alpha = min(0.995, max(0.8, smooth))
        if self.residual:
            # As a post-filter behind the neural net we only remove what is left.
            self.floor_db = -18.0
        else:
            self.floor_db = -40.0

    # -- noise profile learning ------------------------------------------------
    def start_learning(self, seconds: float = 2.0) -> None:
        # (called from the control thread: set the counter last)
        self._learn_acc = np.zeros(self.bins)
        self._learn_n = 0
        self._learning = int(seconds * SAMPLE_RATE / self.hop)

    def clear_learned(self) -> None:
        self.learned = None

    @property
    def learning(self) -> bool:
        return self._learning > 0

    def speech_prob(self) -> float | None:
        return self._sp

    # -- core --------------------------------------------------------------------
    def _frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame * self.win)
        pw = spec.real * spec.real + spec.imag * spec.imag + 1e-12

        if self._learning > 0:
            self._learn_acc += pw
            self._learn_n += 1
            self._learning -= 1
            if self._learning == 0 and self._learn_n:
                self.learned = self._learn_acc / self._learn_n

        # --- noise PSD: unbiased MMSE estimator with speech-presence probability
        if self.noise is None:
            self.noise = pw.copy()
        xi_h1 = 31.62  # 15 dB fixed a-priori SNR under speech presence
        g = xi_h1 / (1.0 + xi_h1)
        post = pw / self.noise
        p = 1.0 / (1.0 + (1.0 + xi_h1) * np.exp(-np.minimum(post * g, 50.0)))
        self.spp_smooth = 0.9 * self.spp_smooth + 0.1 * p
        p = np.where(self.spp_smooth > 0.99, np.minimum(p, 0.99), p)
        est = (1.0 - p) * pw + p * self.noise
        alpha = 0.8 if self.frames > 20 else 0.5
        self.noise = alpha * self.noise + (1.0 - alpha) * est
        self.frames += 1

        lam = self.noise
        if self.learned is not None and self.noise_mode != "adaptive":
            lam = self.learned if self.noise_mode == "learned" else np.maximum(lam, self.learned)
        lam = lam * self.aggressiveness

        # --- decision-directed a-priori SNR + MMSE log-spectral amplitude gain
        gamma = np.minimum(pw / lam, 1e4)
        xi = self.dd_alpha * self.prev_clean_pow / lam + (1 - self.dd_alpha) * np.maximum(
            gamma - 1.0, 0.0
        )
        xi = np.maximum(xi, 10 ** (-25 / 10))
        v = np.maximum(xi * gamma / (1.0 + xi), 1e-8)
        g_lsa = xi / (1.0 + xi) * np.exp(0.5 * exp1(v))
        g_lsa = np.minimum(g_lsa, 1.0)

        # --- OM-LSA: speech-presence-weighted geometric gain with floor. The
        # presence probability is the (bias-compensated) SPP from the tracker above.
        sp = p
        gmin = 10 ** (self.floor_db / 20)
        gain = g_lsa ** sp * gmin ** (1.0 - sp)
        # light frequency smoothing to suppress musical noise
        gain[1:-1] = 0.25 * gain[:-2] + 0.5 * gain[1:-1] + 0.25 * gain[2:]
        self.prev_clean_pow = (g_lsa * g_lsa) * pw
        band = slice(int(300 * self.fft / SAMPLE_RATE), int(4000 * self.fft / SAMPLE_RATE))
        self._sp = float(np.mean(sp[band]))

        out = np.fft.irfft(spec * gain, n=self.fft).astype(np.float32)
        return out * self.win

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        h = self.hop
        for k in range(0, len(x), h):
            self.inbuf[:-h] = self.inbuf[h:]
            self.inbuf[-h:] = x[k : k + h]
            y = self._frame(self.inbuf)
            self.outbuf += y * self.norm
            out[k : k + h] = self.outbuf[:h]
            self.outbuf[:-h] = self.outbuf[h:]
            self.outbuf[-h:] = 0.0
        return out


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

    def start_learning(self, seconds: float = 2.0) -> None:
        self.post.start_learning(seconds)

    def clear_learned(self) -> None:
        self.post.clear_learned()

    @property
    def learning(self) -> bool:
        return self.post.learning

    @property
    def learned(self):
        return self.post.learned


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

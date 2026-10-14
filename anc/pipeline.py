"""The real-time processing chain.

    input gain -> high-pass -> [engine] -> reduction mix (latency-aligned dry)
               -> voice clarity EQ -> noise gate -> AGC -> output gain -> limiter
               -> bypass crossfade (latency-aligned dry)

``Pipeline.process`` is called from the audio callback with exactly
``pipeline.block`` samples (a multiple of the engine hop) at 48 kHz mono.
Everything is preallocated or O(block); no locks are taken on the hot path –
settings are applied by swapping plain Python attributes (atomic under the GIL).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import dsp
from .engines import SAMPLE_RATE, Engine, HybridEngine, SpectralEngine, create_engine

RATE = SAMPLE_RATE
DRIFT_DB = 4.0          # background change that triggers an automatic re-tune
QUIET_VAD = 0.15        # voice-activity level below which a block counts as noise-only
QUIET_SECONDS = 6.0     # rolling store of recent noise-only audio (for re-tuning)
_UNSET = object()


def reduction_to_atten_db(pct: float) -> float:
    """Reduction % is the share of noise *amplitude* removed: 90 % -> 20 dB."""
    pct = min(100.0, max(0.0, pct))
    if pct >= 100.0:
        return float("inf")
    return -20.0 * math.log10(1.0 - pct / 100.0)


class Analyzer:
    """Ring buffers of recent input/output audio for the visualiser (read off-thread)."""

    def __init__(self, size: int = 8192):
        self.size = size
        self.inp = np.zeros(size, dtype=np.float32)
        self.out = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def write(self, x: np.ndarray, y: np.ndarray) -> None:
        n = len(x)
        p = self.pos
        end = p + n
        if end <= self.size:
            self.inp[p:end] = x
            self.out[p:end] = y
        else:
            k = self.size - p
            self.inp[p:] = x[:k]
            self.inp[: n - k] = x[k:]
            self.out[p:] = y[:k]
            self.out[: n - k] = y[k:]
        self.pos = end % self.size

    def latest(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        p = self.pos
        idx = (np.arange(p - n, p)) % self.size
        return self.inp[idx], self.out[idx]


class Pipeline:
    def __init__(self, settings: dict, engine: Engine | None = None, block_mult: int = 1):
        self.engine = engine or create_engine(settings.get("engine", "neural"))
        self.block = self.engine.hop * max(1, int(block_mult))
        while self.block < 256:  # keep callbacks >= ~5 ms
            self.block += self.engine.hop
        br = RATE / self.block
        self.block_rate = br
        L = self.engine.latency
        self.dry_for_mix = dsp.DelayLine(L)
        self.dry_for_bypass = dsp.DelayLine(L)
        self.dry_pre_notch = dsp.DelayLine(L)
        self.hpf = dsp.SosFilter(None)
        self.notch = dsp.SosFilter(None)
        self.clarity = dsp.SosFilter(None)
        self.gate = dsp.NoiseGate(RATE, self.block)
        self.agc = dsp.Agc(RATE, self.block)
        self.limiter = dsp.Limiter(RATE, self.block)
        self.in_gain = dsp.SmoothedParam(1.0, 30, br)
        self.out_gain = dsp.SmoothedParam(1.0, 30, br)
        self.wet = dsp.SmoothedParam(1.0, 40, br)
        self.bypass_mix = dsp.SmoothedParam(0.0, 15, br)
        self.analyzer = Analyzer()
        self.settings: dict = {}
        self._hpf_hz = None
        self._clarity_db = None
        # metrics (written on the audio thread, read by telemetry)
        self.m_in_rms = 0.0
        self.m_out_rms = 0.0
        self.m_in_peak = 0.0
        self.m_out_peak = 0.0
        self.m_removed_db = 0.0
        self.m_proc_ms = 0.0
        self.m_proc_max_ms = 0.0
        self.m_clip = 0
        self.m_voice = 0.0
        self._removed_smooth = 0.0
        self._vad_floor: float | None = None
        self._vad = 0.0
        self._pending: dict | None = None
        # learned noise profile / capture / drift monitoring
        self.profile: dict | None = None
        self._pending_profile = _UNSET
        self._profile_gain_db = 0.0
        self._ref_bands = None
        self._notch_on = None
        self._cap = None
        self._cap_pos = 0
        self._quiet = np.zeros(int(QUIET_SECONDS * RATE), dtype=np.float32)
        self._quiet_pos = 0
        self._quiet_fill = 0
        self._pre_hist = np.zeros(dsp.BAND_FFT, dtype=np.float32)
        self._qband = None
        self._quiet_since_profile = 0.0
        self._drift_over = 0.0
        self.m_drift_db = 0.0
        self.stale = False
        self.apply_settings(settings)
        self.bypass_mix.snap(1.0 if settings.get("bypass") else 0.0)
        self.wet.snap(self.wet.target)
        self.in_gain.snap(self.in_gain.target)
        self.out_gain.snap(self.out_gain.target)

    @property
    def latency(self) -> int:
        return self.engine.latency

    def request_settings(self, s: dict) -> None:
        """Thread-safe: queue new settings; the audio thread applies them at the next
        block boundary so filters are never swapped mid-block."""
        self._pending = dict(s)

    def apply_settings(self, s: dict) -> None:
        self.settings = dict(s)
        self.in_gain.set(dsp.db_to_lin(float(s.get("input_gain_db", 0.0))))
        self.out_gain.set(dsp.db_to_lin(float(s.get("output_gain_db", 0.0))))
        atten = reduction_to_atten_db(float(s.get("reduction", 100.0)))
        # out = enhanced + lim * (dry - enhanced); lim is the residual-noise amplitude.
        lim = 0.0 if math.isinf(atten) else 10 ** (-atten / 20.0)
        self.wet.set(1.0 - lim)
        self.bypass_mix.set(1.0 if s.get("bypass") else 0.0)
        hpf = float(s.get("highpass_hz", 80.0)) if s.get("highpass_enabled", True) else 0.0
        if hpf != self._hpf_hz:
            self.hpf.set(dsp.highpass_sos(hpf, RATE) if hpf > 0 else None)
            self._hpf_hz = hpf
        cl = float(s.get("clarity_db", 0.0))
        if cl != self._clarity_db:
            # presence peak + gentle air shelf, cascaded into one SOS filter
            parts = [p for p in (dsp.peaking_sos(3200.0, cl, 0.8, RATE),
                                 dsp.high_shelf_sos(9000.0, cl * 0.4, RATE)) if p is not None]
            self.clarity.set(np.vstack(parts) if parts else None)
            self._clarity_db = cl
        self.gate.configure(float(s.get("gate_threshold_db", -55.0)),
                            float(s.get("gate_range_db", 30.0)))
        self.agc.target_db = float(s.get("agc_target_db", -20.0))
        self.limiter.ceiling = dsp.db_to_lin(float(s.get("limiter_ceiling_db", -1.0)))
        self.engine.configure(s)
        if self.profile is not None and (
                float(s.get("input_gain_db", 0.0)) != self._profile_gain_db
                or bool(s.get("hum_notch_enabled", True)) != self._notch_on):
            self._apply_profile()

    # ------------------------------------------------------------ noise profile
    def request_profile(self, profile: dict | None) -> None:
        """Thread-safe: install a learned noise profile at the next block boundary."""
        self._pending_profile = profile

    def set_profile(self, profile: dict | None) -> None:
        self.profile = profile
        self._qband = None
        self._quiet_since_profile = 0.0
        self._drift_over = 0.0
        self.m_drift_db = 0.0
        self.stale = False
        self._ref_bands = np.asarray(profile["bands_db"]) if profile else None
        self._apply_profile()

    def _apply_profile(self) -> None:
        p = self.profile
        s = self.settings
        gain_db = float(s.get("input_gain_db", 0.0))
        self._profile_gain_db = gain_db
        self._notch_on = bool(s.get("hum_notch_enabled", True))
        if p is None:
            self.notch.set(None)
            self._set_learned(None, None)
            return
        from .autotune import notch_sos

        self.notch.set(notch_sos(p.get("tones") or []) if self._notch_on else None)
        # The profile was measured at the capture-time input gain; rescale to now.
        scale = 10 ** ((gain_db - float(p.get("input_gain_db", 0.0))) / 10.0)
        raw = np.asarray(p["psd_raw"], dtype=np.float64) * scale
        post = p.get("psd_post")
        post = np.asarray(post, dtype=np.float64) * scale if post is not None else None
        self._set_learned(raw, post)

    def _set_learned(self, raw, post) -> None:
        eng = self.engine
        if isinstance(eng, SpectralEngine):
            eng.learned = raw
        elif isinstance(eng, HybridEngine):
            eng.post.learned = post

    def start_capture(self, seconds: float) -> None:
        """Record ``seconds`` of conditioned input (after high-pass, before notches)."""
        self._cap_pos = 0
        self._cap = np.zeros(int(seconds * RATE), dtype=np.float32)

    @property
    def capture_progress(self) -> float | None:
        cap = self._cap
        return None if cap is None else min(1.0, self._cap_pos / len(cap))

    def take_capture(self) -> np.ndarray | None:
        cap = self._cap
        if cap is None or self._cap_pos < len(cap):
            return None
        self._cap = None
        return cap

    @property
    def quiet_seconds(self) -> float:
        return self._quiet_fill / RATE

    def take_quiet(self, seconds: float) -> np.ndarray:
        """Most recent noise-only audio (speech pauses), oldest first."""
        n = min(int(seconds * RATE), self._quiet_fill)
        idx = (np.arange(self._quiet_pos - n, self._quiet_pos)) % len(self._quiet)
        self._quiet_fill = 0
        return self._quiet[idx].copy()

    def _track_quiet(self, pre: np.ndarray) -> None:
        """Store noise-only audio and watch the background for changes."""
        n = len(pre)
        q = self._quiet
        p = self._quiet_pos
        end = p + n
        if end <= len(q):
            q[p:end] = pre
        else:
            k = len(q) - p
            q[p:] = pre[:k]
            q[: n - k] = pre[k:]
        self._quiet_pos = end % len(q)
        self._quiet_fill = min(len(q), self._quiet_fill + n)
        if self._ref_bands is None:
            return
        bp = dsp.band_power(self._pre_hist)
        self._qband = bp if self._qband is None else self._qband + 0.02 * (bp - self._qband)
        self._quiet_since_profile += n / RATE
        if self._quiet_since_profile < 1.5:
            return
        offset = self._profile_gain_db - float(self.profile.get("input_gain_db", 0.0))
        diff = 10 * np.log10(self._qband + 1e-20) - (self._ref_bands + offset)
        self.m_drift_db = float(np.mean(np.abs(diff)))
        self._drift_over = self._drift_over + n / RATE if self.m_drift_db > DRIFT_DB else 0.0
        if self._drift_over > 2.0:
            self.stale = True

    def _detect_voice(self, enhanced: np.ndarray) -> bool:
        """Energy VAD on the *denoised* signal against an adaptive floor.

        After suppression the background is far below speech, so a minimum-tracking
        floor + ~12 dB margin is a robust, cheap speech detector."""
        db = 10.0 * math.log10(float(np.dot(enhanced, enhanced)) / len(enhanced) + 1e-12)
        if db > -95.0:  # ignore digital silence (engine start-up, muted input)
            if self._vad_floor is None or db < self._vad_floor:
                self._vad_floor = db if self._vad_floor is None else 0.6 * self._vad_floor + 0.4 * db
            else:
                self._vad_floor += 3.0 / self.block_rate  # floor creeps up ~3 dB/s
        elif self._vad_floor is None:
            self._vad = 0.0
            return False
        margin = db - max(self._vad_floor, -90.0) - 12.0
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, margin / 2.5))))
        if db < -75.0:
            p = 0.0
        c = 0.5 if p > self._vad else 0.08  # fast attack, slow release (hangover)
        self._vad += c * (p - self._vad)
        return self._vad > 0.5

    def process(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        pending = self._pending
        if pending is not None:
            self._pending = None
            self.apply_settings(pending)
        prof = self._pending_profile
        if prof is not _UNSET:
            self._pending_profile = _UNSET
            self.set_profile(prof)
        s = self.settings
        raw = x * self.in_gain.next_ramp(len(x))
        self.m_clip += int(np.count_nonzero(np.abs(raw) >= 0.999))
        pre = self.hpf.process(raw)
        cap = self._cap
        if cap is not None and self._cap_pos < len(cap):
            k = min(len(pre), len(cap) - self._cap_pos)
            cap[self._cap_pos : self._cap_pos + k] = pre[:k]
            self._cap_pos += k
        x = self.notch.process(pre)

        enhanced = self.engine.process(x)
        dry = self.dry_for_mix.process(x)
        w = self.wet.next_ramp(len(x))
        y = dry + w * (enhanced - dry)

        # Noise-removed estimate on the aligned signals.
        e_dry = float(np.dot(dry, dry)) + 1e-12
        e_y = float(np.dot(y, y)) + 1e-12
        removed = 10.0 * math.log10(e_dry / e_y)
        self._removed_smooth = 0.9 * self._removed_smooth + 0.1 * max(0.0, removed)

        voice = self._detect_voice(enhanced)
        pre_d = self.dry_pre_notch.process(pre)  # aligned with `enhanced`
        if len(pre_d) >= dsp.BAND_FFT:
            self._pre_hist = pre_d[-dsp.BAND_FFT:]
        else:
            self._pre_hist = np.concatenate([self._pre_hist[len(pre_d):], pre_d])
        if self._vad < QUIET_VAD:
            self._track_quiet(pre_d)
        if self._clarity_db:
            y = self.clarity.process(y)
        if s.get("gate_enabled"):
            y = self.gate.process(y)
            voice = self.gate.open
        if s.get("agc_enabled"):
            y = self.agc.process(y, voice)
        y = y * self.out_gain.next_ramp(len(y))
        if s.get("limiter_enabled", True):
            y = self.limiter.process(y)

        bp = self.bypass_mix.next_ramp(len(y))
        if bp[0] > 0.0 or bp[-1] > 0.0:
            ref = self.dry_for_bypass.process(raw) * self.out_gain.value
            y = y + bp * (ref - y)
        else:
            self.dry_for_bypass.process(raw)

        if s.get("mute_output"):
            out = np.zeros_like(y)
        else:
            out = y
        out = out.astype(np.float32, copy=False)

        # metrics
        self.analyzer.write(dry, out)
        self.m_in_rms = float(np.sqrt(e_dry / len(dry)))
        self.m_out_rms = float(np.sqrt(np.mean(out * out)))
        self.m_in_peak = float(np.max(np.abs(dry)))
        self.m_out_peak = float(np.max(np.abs(out)))
        self.m_removed_db = self._removed_smooth
        self.m_voice = self._vad
        dt = (time.perf_counter() - t0) * 1000.0
        self.m_proc_ms = 0.95 * self.m_proc_ms + 0.05 * dt if self.m_proc_ms else dt
        self.m_proc_max_ms = max(self.m_proc_max_ms * 0.999, dt)
        return out


def process_offline(audio: np.ndarray, settings: dict, engine: Engine | None = None) -> np.ndarray:
    """Run the full real-time chain over a whole mono 48 kHz signal (latency removed)."""
    pipe = Pipeline(settings, engine=engine, block_mult=4)
    n = len(audio)
    lat = pipe.latency
    total = n + lat
    pad = (-total) % pipe.block
    x = np.concatenate([audio.astype(np.float32), np.zeros(lat + pad, dtype=np.float32)])
    out = np.empty_like(x)
    for k in range(0, len(x), pipe.block):
        out[k : k + pipe.block] = pipe.process(x[k : k + pipe.block])
    return out[lat : lat + n]

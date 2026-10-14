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
from .engines import SAMPLE_RATE, Engine, create_engine

RATE = SAMPLE_RATE


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
        self.hpf = dsp.SosFilter(None)
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
        self._vad_floor = -60.0
        self._vad = 0.0
        self._pending: dict | None = None
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

    def _detect_voice(self, enhanced: np.ndarray) -> bool:
        """Energy VAD on the *denoised* signal against an adaptive floor.

        After suppression the background is far below speech, so a minimum-tracking
        floor + ~12 dB margin is a robust, cheap speech detector."""
        db = 10.0 * math.log10(float(np.dot(enhanced, enhanced)) / len(enhanced) + 1e-12)
        if db < self._vad_floor:
            self._vad_floor = 0.6 * self._vad_floor + 0.4 * db
        else:
            self._vad_floor += 2.0 / self.block_rate  # floor creeps up ~2 dB/s
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
        s = self.settings
        raw = x * self.in_gain.next_ramp(len(x))
        self.m_clip += int(np.count_nonzero(np.abs(raw) >= 0.999))
        x = self.hpf.process(raw)

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

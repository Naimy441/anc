"""Low-level, allocation-light DSP building blocks used by the real-time pipeline.

Everything here is mono float32 at the pipeline sample rate and operates on whole
blocks so the hot path stays inside numpy / scipy C code.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, floor: float = -120.0) -> float:
    if x <= 0.0:
        return floor
    return max(floor, 20.0 * math.log10(x))


def time_coeff(ms: float, rate: float) -> float:
    """One-pole smoothing coefficient for a time constant `ms` at `rate` updates/s."""
    if ms <= 0:
        return 0.0
    return float(math.exp(-1.0 / (ms * 1e-3 * rate)))


class Fifo:
    """Preallocated single-producer/single-consumer sample FIFO."""

    def __init__(self, capacity: int):
        self.buf = np.zeros(int(capacity), dtype=np.float32)
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def clear(self) -> None:
        self.n = 0

    def push(self, x: np.ndarray) -> None:
        k = len(x)
        if self.n + k > len(self.buf):
            # Grow (rare: only on misconfiguration / large device blocks).
            nb = np.zeros(max(len(self.buf) * 2, self.n + k), dtype=np.float32)
            nb[: self.n] = self.buf[: self.n]
            self.buf = nb
        self.buf[self.n : self.n + k] = x
        self.n += k

    def pop(self, k: int) -> np.ndarray:
        out = self.buf[:k].copy()
        rest = self.n - k
        if rest > 0:
            self.buf[:rest] = self.buf[k : self.n]
        self.n = rest
        return out

    def drop(self, k: int) -> None:
        k = min(k, self.n)
        rest = self.n - k
        if rest > 0:
            self.buf[:rest] = self.buf[k : self.n]
        self.n = rest


class DelayLine:
    """Fixed integer delay (used to time-align the dry path with an engine)."""

    def __init__(self, delay: int):
        self.delay = int(delay)
        self.mem = np.zeros(self.delay, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.delay == 0:
            return x.copy()
        joined = np.concatenate([self.mem, x])
        self.mem = joined[len(x) :].copy()
        return joined[: len(x)]


class SosFilter:
    """Stateful second-order-sections IIR filter."""

    def __init__(self, sos: np.ndarray | None):
        self.set(sos)

    def set(self, sos: np.ndarray | None) -> None:
        if sos is None:
            self.sos = None
            self.zi = None
            return
        new_sos = np.asarray(sos, dtype=np.float64)
        if self.__dict__.get("zi") is None or self.zi.shape[0] != new_sos.shape[0]:
            self.zi = np.zeros((new_sos.shape[0], 2), dtype=np.float64)
        self.sos = new_sos

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.sos is None:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32, copy=False)


def highpass_sos(cutoff_hz: float, rate: int, order: int = 4) -> np.ndarray | None:
    if cutoff_hz <= 0:
        return None
    return butter(order, cutoff_hz, btype="highpass", fs=rate, output="sos")


def peaking_sos(freq: float, gain_db: float, q: float, rate: int) -> np.ndarray | None:
    """RBJ-cookbook peaking EQ as a single SOS row."""
    if abs(gain_db) < 1e-3:
        return None
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / rate
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b0, b1, b2 = 1 + alpha * a, -2 * cw, 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * cw, 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def high_shelf_sos(freq: float, gain_db: float, rate: int, s: float = 0.8) -> np.ndarray | None:
    if abs(gain_db) < 1e-3:
        return None
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * freq / rate
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / 2 * math.sqrt((a + 1 / a) * (1 / s - 1) + 2)
    sa = 2 * math.sqrt(a) * alpha
    b0 = a * ((a + 1) + (a - 1) * cw + sa)
    b1 = -2 * a * ((a - 1) + (a + 1) * cw)
    b2 = a * ((a + 1) + (a - 1) * cw - sa)
    a0 = (a + 1) - (a - 1) * cw + sa
    a1 = 2 * ((a - 1) - (a + 1) * cw)
    a2 = (a + 1) - (a - 1) * cw - sa
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def ramp(start: float, end: float, n: int) -> np.ndarray:
    """Per-sample linear gain ramp (click-free parameter changes)."""
    if start == end:
        return np.full(n, start, dtype=np.float32)
    return np.linspace(start, end, n, endpoint=False, dtype=np.float32) + np.float32(
        (end - start) / n
    )


class SmoothedParam:
    """A block-rate parameter that glides to its target to avoid zipper noise."""

    def __init__(self, value: float, glide_ms: float, block_rate: float):
        self.value = float(value)
        self.target = float(value)
        self.coeff = time_coeff(glide_ms, block_rate)

    def set(self, target: float) -> None:
        self.target = float(target)

    def snap(self, value: float) -> None:
        self.value = self.target = float(value)

    def next_ramp(self, n: int) -> np.ndarray:
        start = self.value
        self.value = self.target + (self.value - self.target) * self.coeff
        if abs(self.value - self.target) < 1e-5:
            self.value = self.target
        return ramp(start, self.value, n)


class NoiseGate:
    """Downward expander with hysteresis + hold, driven by a block envelope."""

    def __init__(self, rate: int, block: int):
        self.rate = rate
        self.block = block
        self.gain = 1.0
        self.env = 0.0
        self.hold_left = 0
        self.open = True
        self.configure(-55.0, 30.0)

    def configure(self, threshold_db: float, range_db: float, attack_ms: float = 3.0,
                  release_ms: float = 140.0, hold_ms: float = 120.0) -> None:
        self.threshold = db_to_lin(threshold_db)
        self.close_threshold = db_to_lin(threshold_db - 4.0)  # 4 dB hysteresis
        self.floor = db_to_lin(-abs(range_db))
        br = self.rate / self.block
        self.att = time_coeff(attack_ms, br)
        self.rel = time_coeff(release_ms, br)
        self.env_coeff = time_coeff(10.0, br)
        self.hold_blocks = int(hold_ms * 1e-3 * br)

    def process(self, x: np.ndarray, key: np.ndarray | None = None) -> np.ndarray:
        k = x if key is None else key
        level = float(np.sqrt(np.mean(k * k) + 1e-20))
        self.env = max(level, self.env * self.env_coeff + level * (1 - self.env_coeff))
        if self.env >= self.threshold:
            self.open = True
            self.hold_left = self.hold_blocks
        elif self.env < self.close_threshold:
            if self.hold_left > 0:
                self.hold_left -= 1
            else:
                self.open = False
        target = 1.0 if self.open else self.floor
        c = self.att if target > self.gain else self.rel
        start = self.gain
        self.gain = target + (self.gain - target) * c
        return x * ramp(start, self.gain, len(x))


class Agc:
    """Slow automatic gain control that only adapts while voice is present."""

    def __init__(self, rate: int, block: int):
        br = rate / block
        self.up = time_coeff(1800.0, br)
        self.down = time_coeff(350.0, br)
        self.gain_db = 0.0
        self.target_db = -20.0
        self.max_gain_db = 18.0

    def process(self, x: np.ndarray, voice: bool) -> np.ndarray:
        start = db_to_lin(self.gain_db)
        if voice:
            level_db = lin_to_db(float(np.sqrt(np.mean(x * x) + 1e-20)))
            want = float(np.clip(self.target_db - level_db, -18.0, self.max_gain_db))
            c = self.up if want > self.gain_db else self.down
            self.gain_db = want + (self.gain_db - want) * c
        return x * ramp(start, db_to_lin(self.gain_db), len(x))


class Limiter:
    """Zero-latency block limiter: instant attack on the block peak, smooth release,
    followed by a soft clipper that guarantees the ceiling."""

    def __init__(self, rate: int, block: int, ceiling_db: float = -1.0):
        self.ceiling = db_to_lin(ceiling_db)
        self.rel = time_coeff(80.0, rate / block)
        self.gain = 1.0
        self.reduction_db = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        need = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
        start = self.gain
        if need < self.gain:
            self.gain = need
            start = need  # instant attack for this block
        else:
            self.gain = need + (self.gain - need) * self.rel
        y = x * ramp(start, self.gain, len(x))
        self.reduction_db = -lin_to_db(self.gain)
        # Soft-knee safety clip above 90 % of ceiling.
        c = self.ceiling
        knee = 0.9 * c
        over = np.abs(y) > knee
        if over.any():
            s = np.sign(y[over])
            a = np.abs(y[over]) - knee
            y[over] = s * (knee + (c - knee) * np.tanh(a / (c - knee)))
        return y


class StreamResampler:
    """Streaming band-limited resampler (Kaiser-windowed sinc, polyphase table with
    linear phase interpolation).

    Unlike block-based resamplers it emits output in exact proportion to its input
    (no bursts), has only ``taps/2`` samples of delay, and its ratio can be nudged
    on the fly – which is what clock-drift compensation between two independent
    audio devices needs.
    """

    def __init__(self, in_rate: float, out_rate: float, taps: int = 32, phases: int = 256,
                 beta: float = 8.0):
        self.taps = taps
        self.half = taps // 2
        self.phases = phases
        self.nominal = float(in_rate) / float(out_rate)  # input samples per output sample
        self.ratio = self.nominal
        cutoff = min(1.0, out_rate / in_rate) * 0.92
        offs = np.arange(-self.half + 1, self.half + 1)  # tap offsets relative to floor(t)
        frac = np.arange(phases + 1)[:, None] / phases
        x = offs[None, :] - frac  # distance of each tap from the read position
        win = np.kaiser(2 * taps + 1, beta)
        w = np.interp(x, np.linspace(-self.half, self.half, 2 * taps + 1), win)
        table = cutoff * np.sinc(cutoff * x) * w
        table /= table.sum(axis=1, keepdims=True)  # unity DC gain on every phase
        self.table = table.astype(np.float32)
        self.offs = offs
        self.buf = np.zeros(taps, dtype=np.float32)
        self.t = float(self.half - 1)  # read position inside buf

    @property
    def delay(self) -> float:
        """Group delay in input samples."""
        return float(self.half)

    def set_ratio_adjust(self, adj: float) -> None:
        """Relative speed tweak: adj > 0 consumes input faster (fewer outputs)."""
        self.ratio = self.nominal * (1.0 + adj)

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.buf, x.astype(np.float32, copy=False)])
        last = len(buf) - 1 - self.half  # max read position with full support
        if self.t > last:
            self.buf = buf
            return np.zeros(0, dtype=np.float32)
        n = int((last - self.t) // self.ratio) + 1
        pos = self.t + self.ratio * np.arange(n)
        base = np.floor(pos).astype(np.int64)
        ph = (pos - base) * self.phases
        p0 = ph.astype(np.int64)
        a = (ph - p0).astype(np.float32)[:, None]
        coefs = self.table[p0] * (1.0 - a) + self.table[p0 + 1] * a
        frames = buf[base[:, None] + self.offs[None, :]]
        y = np.einsum("ij,ij->i", frames, coefs).astype(np.float32)
        t_next = self.t + self.ratio * n
        keep_from = max(0, int(np.floor(t_next)) - self.half + 1)
        self.buf = buf[keep_from:].copy()
        self.t = t_next - keep_from
        return y

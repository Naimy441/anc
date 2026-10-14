"""Acoustic feedback (howl) guard.

When the processed output reaches the microphone again (speakers, open earbuds,
a headset mic next to its speaker), the loop can ring at whatever frequency has
the most loop gain – typically a piercing tone around 2–6 kHz.

Detection runs on the final output, frame by frame (2048-point FFT, one frame per
block), using the classic howling-detection features (van Waterschoot & Moonen,
"Fifty years of acoustic feedback control", 2011):

* PNPR – peak-to-neighbouring power ratio: a howl is a single sharp line far
  above the spectrum around it.
* PHPR – peak-to-harmonic power ratio: a howl is a pure tone, while voices and
  instruments always carry harmonics (and the peak is not itself a harmonic of
  a stronger lower line).
* Persistence – the line stays within ±1 bin for a sustained time and does not
  decay (voices glide in pitch and fade).

A confirmed howl gets a narrow peaking cut (Q≈18) at the refined frequency,
starting at –15 dB and deepening by 8 dB (to –42 dB) each time it is re-detected.
The output is also ducked by 6 dB for a moment and the AGC is frozen so it cannot
push the loop gain back up. Cuts that have not re-triggered for a while are
released slowly, so the guard does not permanently colour the sound.
"""
from __future__ import annotations

import numpy as np

from . import dsp

SENSITIVITY = {
    #         PNPR dB  persistence ms  PHPR dB  min level dBFS
    "low": (20.0, 300.0, 14.0, -45.0),
    "normal": (15.0, 180.0, 12.0, -55.0),
    "high": (12.0, 110.0, 10.0, -62.0),
}

FIRST_DEPTH_DB = -15.0
STEP_DB = -8.0
MAX_DEPTH_DB = -42.0
NOTCH_Q = 18.0
MAX_NOTCHES = 8
RELEASE_AFTER_S = 45.0
RELEASE_DB_PER_S = 3.0
# Last resort when narrow cuts cannot keep up (speaker right next to the mic, loop
# gain > 1 across a whole band): pull the overall output down in steps.
ESCALATE_EVENTS = 4         # this many squeals ...
ESCALATE_WINDOW_S = 5.0     # ... within this window
SAFETY_STEP_DB = -6.0
SAFETY_MAX_DB = -24.0
SAFETY_HOLD_S = 8.0
SAFETY_RECOVER_DB_PER_S = 1.5


class _Notch:
    def __init__(self, freq: float, depth: float, now: float, rate: int):
        self.freq = freq
        self.depth = depth
        self.target = depth
        self.last_hit = now
        self.rate = rate
        self.filt = dsp.SosFilter(None)
        self._applied = None
        self.update()

    def update(self) -> None:
        if self._applied is None or abs(self.depth - self._applied) > 0.05:
            self.filt.set(dsp.peaking_sos(self.freq, self.depth, NOTCH_Q, self.rate))
            self._applied = self.depth


class FeedbackGuard:
    def __init__(self, rate: int = 48000, block: int = 512, fft: int = 2048):
        self.rate = rate
        self.block = block
        self.fft = fft
        self.win = np.hanning(fft).astype(np.float32)
        self.amp_scale = 4.0 / fft  # |X| of a Hann-windowed sine -> amplitude
        self.df = rate / fft
        self.lo = int(350 / self.df)
        self.low_bin = int(1000 / self.df)  # below this, demand twice the persistence
        self.hi = min(fft // 2 - 16, int(14000 / self.df))
        self.buf = np.zeros(fft, dtype=np.float32)
        self.notches: list[_Notch] = []
        self.tracks: list[dict] = []
        self.time = 0.0
        self.events = 0
        self.last_event: dict | None = None
        self.duck = dsp.SmoothedParam(1.0, 600.0, rate / block)
        self.agc_hold_until = 0.0
        self.safety_db = 0.0
        self._event_times: list[float] = []
        self._safety = dsp.SmoothedParam(1.0, 80.0, rate / block)
        self.configure("normal")

    def configure(self, sensitivity: str = "normal") -> None:
        self.pnpr, persist_ms, self.phpr, self.min_db = SENSITIVITY.get(
            sensitivity, SENSITIVITY["normal"])
        self.persist_s = persist_ms / 1000.0

    @property
    def agc_frozen(self) -> bool:
        return self.time < self.agc_hold_until

    def reset(self) -> None:
        self.notches = []
        self.tracks = []
        self.buf[:] = 0.0
        self.safety_db = 0.0
        self._event_times = []

    def active(self) -> list[dict]:
        return [{"freq": round(n.freq, 1), "depth": round(n.depth, 1)} for n in self.notches]

    # ------------------------------------------------------------------ audio
    def process(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        dt = n / self.rate
        self.time += dt
        self._update_notches(dt)
        for nt in self.notches:
            y = nt.filt.process(y)
        if self.duck.value < 0.999 or self.duck.target < 1.0:
            y = y * self.duck.next_ramp(n)
            self.duck.set(1.0)
        last = self._event_times[-1] if self._event_times else -1e9
        if self.safety_db < 0.0 and self.time - last > SAFETY_HOLD_S:
            self.safety_db = min(0.0, self.safety_db + SAFETY_RECOVER_DB_PER_S * dt)
        self._safety.set(dsp.db_to_lin(self.safety_db))
        if self._safety.value < 0.9999 or self.safety_db < 0.0:
            y = y * self._safety.next_ramp(n)
        self._analyze(y, dt)
        return y

    def _update_notches(self, dt: float) -> None:
        keep = []
        for nt in self.notches:
            if self.time - nt.last_hit > RELEASE_AFTER_S:
                nt.target = 0.0
            if nt.depth < nt.target:
                nt.depth = min(nt.target, nt.depth + RELEASE_DB_PER_S * dt)
            if nt.target == 0.0 and nt.depth > -0.3:
                continue  # fully released
            nt.update()
            keep.append(nt)
        self.notches = keep

    def _analyze(self, y: np.ndarray, dt: float) -> None:
        n = len(y)
        if n >= self.fft:
            self.buf[:] = y[-self.fft:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = y
        spec = np.fft.rfft(self.buf * self.win)
        mag = np.abs(spec) * self.amp_scale
        pdb = 20.0 * np.log10(mag + 1e-9)
        band = pdb[self.lo : self.hi]
        # local maxima
        is_max = (band[1:-1] > band[:-2]) & (band[1:-1] >= band[2:])
        idx = np.nonzero(is_max)[0] + 1 + self.lo
        hits = []
        if len(idx):
            top = idx[np.argsort(pdb[idx])[-4:]]
            for k in top:
                if pdb[k] < self.min_db:
                    continue
                nb = np.concatenate([pdb[max(0, k - 14) : k - 3], pdb[k + 4 : k + 15]])
                if len(nb) < 8 or pdb[k] - np.median(nb) < self.pnpr:
                    continue
                if not self._is_pure(pdb, k):
                    continue
                hits.append(k)
        self._track(hits, pdb, dt)

    def _is_pure(self, pdb: np.ndarray, k: int) -> bool:
        """Reject lines with harmonics (voices, instruments)."""
        top = pdb[k]
        for mult in (2.0, 3.0, 0.5, 1.0 / 3.0):
            h = int(round(k * mult))
            if h < 4 or h >= len(pdb) - 2:
                continue
            if top - np.max(pdb[h - 1 : h + 2]) < self.phpr:
                return False
        return True

    def _refine(self, pdb: np.ndarray, k: int) -> float:
        a, b, c = pdb[k - 1], pdb[k], pdb[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
        return (k + float(np.clip(frac, -0.5, 0.5))) * self.df

    def _track(self, hits: list[int], pdb: np.ndarray, dt: float) -> None:
        for tr in self.tracks:
            tr["seen"] = False
        for k in hits:
            tr = next((t for t in self.tracks if abs(t["bin"] - k) <= 1), None)
            if tr is None:
                tr = {"bin": k, "age": 0.0, "start_db": float(pdb[k]), "miss": 0.0, "freqs": [],
                      "dbs": []}
                self.tracks.append(tr)
            tr["bin"] = k
            tr["age"] += dt
            tr["seen"] = True
            tr["miss"] = 0.0
            tr["db"] = float(pdb[k])
            tr["freqs"].append(self._refine(pdb, k))
            tr["dbs"].append(tr["db"])
            need = self.persist_s * (2.0 if k < self.low_bin else 1.0)
            # A howl's pitch is locked; voices drift (jitter, intonation).
            steady = len(tr["freqs"]) < 3 or float(np.std(tr["freqs"][-12:])) < 0.06 * self.df + 0.002 * tr["freqs"][-1]
            if not steady:
                tr["age"] = 0.0
                tr["freqs"] = tr["freqs"][-3:]
                tr["dbs"] = tr["dbs"][-1:]
                tr["start_db"] = tr["db"]
            elif (tr["age"] >= need and tr["db"] >= tr["start_db"] - 3.0) or self._runaway(tr, dt, need):
                self._howl(k, pdb)
                tr["age"] = 0.0
                tr["freqs"] = []
                tr["dbs"] = []
                tr["start_db"] = tr["db"] - 6.0  # needs to persist again to deepen
        alive = []
        for tr in self.tracks:
            if not tr["seen"]:
                tr["miss"] += dt
                if tr["miss"] > 0.06:  # a howl never disappears for 60 ms
                    continue
            alive.append(tr)
        self.tracks = alive

    @staticmethod
    def _runaway(tr: dict, dt: float, need: float) -> bool:
        """Fast path for a runaway loop: a locked pure tone whose level keeps
        climbing steadily (> 30 dB/s, linear in dB, still rising now). A syllable
        jumps up and plateaus, so it fails the straight-line test."""
        d = tr["dbs"]
        if tr["age"] < 0.5 * need or len(d) < 6:
            return False
        y = np.asarray(d[-16:])
        x = np.arange(len(y)) * dt
        slope, icpt = np.polyfit(x, y, 1)
        resid = y - (slope * x + icpt)
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
        r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot
        recent = (y[-1] - y[-4]) / (3 * dt)
        return slope > 30.0 and r2 > 0.9 and recent > 15.0

    def _howl(self, k: int, pdb: np.ndarray) -> None:
        b = pdb[k]
        freq = self._refine(pdb, k)
        nt = next((x for x in self.notches if abs(x.freq - freq) < max(1.5 * self.df, 0.02 * freq)), None)
        if nt is not None:
            nt.target = max(MAX_DEPTH_DB, min(nt.depth, nt.target) + STEP_DB)
            nt.depth = nt.target
            nt.freq = freq
            nt._applied = None
            nt.last_hit = self.time
            nt.update()
        else:
            if len(self.notches) >= MAX_NOTCHES:
                self.notches.remove(min(self.notches, key=lambda x: x.last_hit))
            nt = _Notch(freq, FIRST_DEPTH_DB, self.time, self.rate)
            self.notches.append(nt)
        self.events += 1
        self._event_times = [t for t in self._event_times if self.time - t < ESCALATE_WINDOW_S]
        self._event_times.append(self.time)
        if len(self._event_times) >= ESCALATE_EVENTS:
            self.safety_db = max(SAFETY_MAX_DB, self.safety_db + SAFETY_STEP_DB)
        self.last_event = {"freq": round(freq, 1), "depth": round(nt.depth, 1), "t": round(self.time, 3),
                           "safety_db": round(self.safety_db, 1),
                           "n": self.events, "level_db": round(float(b), 1)}
        self.duck.value = min(self.duck.value, 0.5)  # instant -6 dB
        self.duck.set(1.0)
        self.agc_hold_until = self.time + 5.0


def describe(freq: float) -> str:
    return f"{freq / 1000:.2f} kHz" if freq >= 1000 else f"{freq:.0f} Hz"


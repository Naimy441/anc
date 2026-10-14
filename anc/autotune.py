"""Learn-and-tune: measure the background noise, then find the strongest suppression
that does not add artifacts or damage speech.

Given a few seconds of noise-only audio captured from the live input, ``analyze``:

1. Detects stationary tonal components (fan blade-pass hum, mains hum, coil whine)
   and designs narrow notch filters for them.
2. Splits the capture in two: the first half trains the noise profile, the second
   half is unseen test noise (so scores are honest, not self-fitted).
3. Builds a test mixture: the test noise + a synthetic voice at 5 dB SNR.
4. Runs every candidate configuration through the *runtime* engines:
     * Spectral  – learned-profile OM-LSA, a grid of over-subtraction × gain floor
     * Neural    – DeepFilterNet3 at several reduction amounts
     * Hybrid    – DeepFilterNet3 + learned-residual OM-LSA grid
5. Scores each candidate on
     * noise removed (dB) on the unseen noise,
     * added artifacts: log-kurtosis ratio of the residual's power spectrum
       (the standard musical-noise measure, Uemura et al. 2012) – 0 means the
       leftover noise is just a quieter copy of the original, larger values mean
       chirpy "musical" artifacts,
     * voice quality: SI-SDR of the processed mixture against the clean voice.
6. Keeps candidates whose artifacts are under the style's limit and whose voice
   quality is within the style's tolerance of the best candidate, and picks the one
   that removes the most noise (ties -> lower latency).
"""
from __future__ import annotations

import copy
import math
import time
from typing import Callable

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, iirnotch, sosfilt, tf2sos, welch

from . import dsp
from .engines import SAMPLE_RATE, DeepFilterNetEngine, SpectralEngine

RATE = SAMPLE_RATE

STYLES = {
    #            voice-quality tolerance (dB)   max artifact (log kurtosis ratio)
    "gentle": {"voice_tol": 0.5, "artifact_max": 0.15},
    "balanced": {"voice_tol": 1.5, "artifact_max": 0.35},
    "max": {"voice_tol": 3.0, "artifact_max": 0.7},
}

SPECTRAL_BETAS = [1.0, 1.4, 2.0, 2.8]
SPECTRAL_FLOORS = [-15.0, -20.0, -25.0, -30.0, -35.0, -40.0]
NEURAL_REDUCTIONS = [100.0, 99.7, 99.0, 97.0, 94.0, 90.0]
POST_BETAS = [1.0, 1.5, 2.2]
POST_FLOORS = [-10.0, -15.0, -20.0, -25.0]
TEST_SNR_DB = 5.0
INAUDIBLE_DBFS = -72.0  # residual below this level cannot be heard as artifacts
WARMUP = int(0.4 * RATE)

LATENCY_MS = {"spectral": 16.0, "neural": 32.0, "hybrid": 48.0}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

_VOWELS = [  # F1, F2, F3 (Hz) – a, e, i, o, u, schwa
    (730, 1090, 2440), (530, 1840, 2480), (270, 2290, 3010),
    (570, 840, 2410), (300, 870, 2240), (500, 1500, 2500),
]


def synth_voice(n: int, rate: int = RATE, seed: int = 7) -> np.ndarray:
    """Deterministic speech stand-in: formant-filtered glottal harmonics with pitch
    contours, syllable timing, pauses and fricatives. Used only to *measure* voice
    damage during tuning (never played)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / rate
    # syllable plan
    env = np.zeros(n)
    f1 = np.full(n, 500.0)
    f2 = np.full(n, 1500.0)
    f3 = np.full(n, 2500.0)
    fric = np.zeros(n)
    pos = int(0.05 * rate)
    while pos < n:
        dur = int(rng.uniform(0.12, 0.30) * rate)
        a, b = pos, min(n, pos + dur)
        v = _VOWELS[rng.integers(len(_VOWELS))]
        ramp = np.sin(np.linspace(0, np.pi, b - a)) ** 0.6
        env[a:b] = np.maximum(env[a:b], ramp * rng.uniform(0.6, 1.0))
        f1[a:b], f2[a:b], f3[a:b] = v
        if rng.random() < 0.35:  # fricative onset
            fd = int(rng.uniform(0.05, 0.11) * rate)
            fa = max(0, a - fd)
            fric[fa:a] = np.sin(np.linspace(0, np.pi, a - fa)) * rng.uniform(0.25, 0.5)
        gap = rng.uniform(0.03, 0.12) if rng.random() < 0.8 else rng.uniform(0.25, 0.5)
        pos = b + int(gap * rate)
    # smooth formant transitions
    k = np.ones(int(0.03 * rate)) / int(0.03 * rate)
    f1, f2, f3 = (np.convolve(f, k, "same") for f in (f1, f2, f3))
    # pitch: declination + intonation + jitter
    f0 = 125 * (1 + 0.15 * np.sin(2 * np.pi * 0.9 * t + 1.0)) * (1 - 0.05 * (t % 2.0))
    f0 *= 1 + 0.004 * np.convolve(rng.standard_normal(n), np.ones(200) / 200 * 14, "same")
    phase = 2 * np.pi * np.cumsum(f0) / rate
    voiced = np.zeros(n)
    kmax = int(7000 / f0.min())
    for h in range(1, kmax + 1):
        fh = h * f0
        amp = (1.0 / h) * (fh < 7000)
        res = sum(1.0 / (1.0 + ((fh - fc) / bw) ** 2) for fc, bw in ((f1, 90), (f2, 120), (f3, 170)))
        voiced += amp * (0.08 + res) * np.sin(h * phase)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    fr = np.fft.rfftfreq(n, 1 / rate)
    spec *= ((fr > 2500) & (fr < 9000)).astype(float)
    hiss = np.fft.irfft(spec, n)
    y = voiced * env + 3.0 * hiss * fric
    return (y / (np.sqrt(np.mean(y ** 2)) + 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def detect_tones(x: np.ndarray, rate: int = RATE, max_tones: int = 8) -> list[dict]:
    """Stationary narrowband peaks (hum) between 40 Hz and 4 kHz."""
    nper = 16384 if len(x) >= 16384 * 2 else 8192
    if len(x) < nper:
        return []
    f, p = welch(x, rate, nperseg=nper, noverlap=nper // 2)
    db = 10 * np.log10(p + 1e-20)
    df = f[1] - f[0]
    base = median_filter(db, size=max(9, int(80 / df) | 1), mode="nearest")
    prom = db - base
    lo, hi = np.searchsorted(f, 40), np.searchsorted(f, 4000)
    peaks, props = find_peaks(prom[:hi], height=10.0, distance=max(1, int(12 / df)))
    peaks = [k for k in peaks if k >= lo]
    if not peaks:
        return []
    # Stationarity: the peak must be present in every quarter of the capture.
    quarters = np.array_split(x, 4)
    qprom = []
    for q in quarters:
        if len(q) < nper // 2:
            continue
        fq, pq = welch(q, rate, nperseg=nper // 2, noverlap=nper // 4)
        dq = 10 * np.log10(pq + 1e-20)
        bq = median_filter(dq, size=max(9, int(80 / (fq[1] - fq[0])) | 1), mode="nearest")
        qprom.append(np.interp(f, fq, dq - bq))
    tones = []
    for k in peaks:
        # -3 dB width of the peak
        lo_k, hi_k = k, k
        while lo_k > 0 and db[lo_k - 1] > db[k] - 3.0:
            lo_k -= 1
        while hi_k < len(db) - 1 and db[hi_k + 1] > db[k] - 3.0:
            hi_k += 1
        w = (hi_k - lo_k + 1) * df
        if qprom and min(q[k] for q in qprom) < 6.0:
            continue
        # parabolic interpolation of the log peak -> sub-bin frequency accuracy
        a, b_, c = db[k - 1], db[k], db[k + 1]
        den = a - 2 * b_ + c
        frac = 0.5 * (a - c) / den if den != 0 else 0.0
        tones.append({"freq": round(float(f[k] + np.clip(frac, -0.5, 0.5) * df), 1), "prominence_db": round(float(prom[k]), 1),
                      "bandwidth": round(float(min(25.0, max(3.0, 1.5 * w))), 1)})
    tones.sort(key=lambda t: -t["prominence_db"])
    return sorted(tones[:max_tones], key=lambda t: t["freq"])


def select_noise_only(x: np.ndarray, rate: int = RATE) -> tuple[np.ndarray, float]:
    """Drop 100 ms segments that are clearly louder than the background (a cough,
    a word, a click) so they are not learned as noise. Returns (audio, seconds cut)."""
    n = int(0.1 * rate)
    m = len(x) // n
    if m < 10:
        return x, 0.0
    seg = x[: m * n].reshape(m, n).astype(np.float64)
    lv = 10 * np.log10(np.mean(seg ** 2, axis=1) + 1e-20)
    keep = lv <= np.percentile(lv, 30) + 4.0
    # also drop the neighbours of loud segments (onsets/decays)
    loud = ~keep
    keep &= ~(np.roll(loud, 1) | np.roll(loud, -1))
    if keep.sum() * n < int(1.5 * rate):
        return x, 0.0
    return seg[keep].ravel().astype(np.float32), float((m - keep.sum()) * n / rate)


def notch_sos(tones: list[dict], rate: int = RATE) -> np.ndarray | None:
    rows = []
    for t in tones:
        q = t["freq"] / t["bandwidth"]
        b, a = iirnotch(t["freq"], q, fs=rate)
        rows.append(tf2sos(b, a))
    return np.vstack(rows) if rows else None


def stationarity_db(x: np.ndarray, rate: int = RATE) -> float:
    """How much the noise fluctuates: median over 300 Hz–8 kHz bands of the dB std of
    250 ms band levels (Welch-averaged, so a perfectly steady noise reads ~0.5 dB)."""
    n = int(0.25 * rate)
    frames = len(x) // n
    if frames < 4:
        return 0.0
    lv = []
    for i in range(frames):
        seg = x[i * n : (i + 1) * n]
        blocks = [dsp.band_power(seg[j : j + dsp.BAND_FFT])
                  for j in range(0, len(seg) - dsp.BAND_FFT + 1, dsp.BAND_FFT // 2)]
        lv.append(10 * np.log10(np.mean(blocks, axis=0) + 1e-20))
    lv = np.array(lv)[:, dsp.BAND_EDGES[:-1] >= 300]
    return float(np.median(np.std(lv, axis=0)))


def log_kurtosis_ratio(ref: np.ndarray, out: np.ndarray, rate: int = RATE) -> float:
    """ln(kurtosis(out power spectrum) / kurtosis(ref power spectrum)), 150 Hz–8 kHz."""
    def kurt(sig):
        n, h = 512, 256
        m = (len(sig) - n) // h
        if m < 8:
            return 1.0
        idx = np.arange(n)[None, :] + h * np.arange(m)[:, None]
        spec = np.fft.rfft(sig[idx] * np.hanning(n), axis=1)
        p = spec.real ** 2 + spec.imag ** 2
        f = np.fft.rfftfreq(n, 1 / rate)
        p = p[:, (f > 150) & (f < 8000)]
        p = p / (np.mean(p, axis=0, keepdims=True) + 1e-20)
        p = p.ravel()
        mu = p.mean()
        var = np.mean((p - mu) ** 2)
        return float(np.mean((p - mu) ** 4) / (var * var + 1e-30))

    return float(math.log(max(kurt(out), 1e-6) / max(kurt(ref), 1e-6)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    a = float(np.dot(est, ref) / (np.dot(ref, ref) + 1e-20))
    t = a * ref
    return float(10 * np.log10(np.sum(t * t) / (np.sum((est - t) ** 2) + 1e-20) + 1e-20))


def _db(x: np.ndarray) -> float:
    return 10 * math.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20)


def _run(engine, x: np.ndarray, block: int = 4096) -> np.ndarray:
    """Process ``x`` and return output aligned with it (engine latency removed)."""
    lat = engine.latency
    pad = (-(len(x) + lat)) % engine.hop
    xx = np.concatenate([x.astype(np.float32), np.zeros(lat + pad, dtype=np.float32)])
    block = max(engine.hop, block // engine.hop * engine.hop)
    outs = [engine.process(xx[k : k + block]) for k in range(0, len(xx), block)]
    y = np.concatenate(outs, axis=-1)
    return y[..., lat : lat + len(x)]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def _score(name, family, params, ref_noise, out_noise, voice, out_mix):
    sl = slice(WARMUP, None)
    removal = _db(ref_noise[sl]) - _db(out_noise[sl])
    resid = _db(out_noise[sl])
    art = 0.0 if resid < INAUDIBLE_DBFS else log_kurtosis_ratio(ref_noise[sl], out_noise[sl])
    return {
        "label": name, "family": family, "params": params,
        "removal_db": round(removal, 2), "artifact": round(max(0.0, art), 3),
        "voice_sdr": round(si_sdr(voice[sl], out_mix[sl]), 2), "residual_dbfs": round(resid, 1),
        "latency_ms": LATENCY_MS[family],
    }


def analyze(noise: np.ndarray, style: str = "balanced", families: list[str] | None = None,
            settings: dict | None = None, progress: Callable[[dict], None] | None = None,
            neural: DeepFilterNetEngine | None = None) -> dict:
    """Learn the noise in ``noise`` (48 kHz mono, noise only) and pick the best setup."""
    t0 = time.perf_counter()
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < int(1.5 * RATE):
        raise ValueError("need at least 1.5 s of noise to learn a profile")
    families = families or ["spectral", "neural", "hybrid"]
    cfg = STYLES.get(style, STYLES["balanced"])
    settings = dict(settings or {})
    emit = progress or (lambda _m: None)

    emit({"stage": "analyzing", "progress": 0.05})
    noise, rejected_s = select_noise_only(noise)
    tones = detect_tones(noise)
    sos = notch_sos(tones)
    clean_in = sosfilt(sos, noise).astype(np.float32) if sos is not None else noise
    # Notch transients settle within ~0.2 s.
    settle = int(0.2 * RATE) if sos is not None else 0
    work = clean_in[settle:]
    half = len(work) // 2
    learn, test = work[:half], work[half:]
    raw_test = noise[settle + half :]
    if len(test) < int(0.8 * RATE):
        learn = test = work
        raw_test = noise[settle:]
    voice = synth_voice(len(test))
    voice *= 10 ** ((_db(test) + TEST_SNR_DB) / 20)
    mix = (test + voice).astype(np.float32)
    level_dbfs = _db(noise)
    stat = stationarity_db(noise)

    cands: list[dict] = []
    total = sum({"spectral": 1, "neural": 2, "hybrid": 1}[f] for f in families) + 1
    done = [1]

    def step(stage, new):
        done[0] += 1
        cands.extend(new)
        emit({"stage": stage, "progress": round(0.1 + 0.85 * done[0] / total, 3),
              "candidates": [_public(c) for c in new]})

    def spectral_batch(engine_input_learn, engine_input_test, engine_input_mix, betas, floors,
                       residual):
        eng = SpectralEngine(residual=residual)
        eng.configure({**settings, "noise_mode": "learned"})
        grid = [(b, f) for b in betas for f in floors]
        eng.set_candidates([g[0] for g in grid], [g[1] for g in grid])
        eng.learned = eng.psd_of(engine_input_learn)
        psd = eng.learned.copy()
        # warm the trackers on the tail of the training audio, then fork the state
        warm = engine_input_learn[-int(0.5 * RATE):]
        _run(eng, warm)
        snap = copy.deepcopy(eng.__dict__)
        out_n = _run(eng, engine_input_test)
        eng.__dict__.update(copy.deepcopy(snap))
        out_m = _run(eng, engine_input_mix)
        return grid, out_n, out_m, psd

    psd_raw = SpectralEngine().psd_of(learn)
    psd_post = None

    if "spectral" in families:
        grid, out_n, out_m, psd_raw = spectral_batch(learn, test, mix, SPECTRAL_BETAS,
                                                     SPECTRAL_FLOORS, residual=False)
        step("spectral", [
            _score(f"Spectral · {b:g}× · {f:g} dB", "spectral",
                   {"engine": "spectral", "spectral_aggressiveness": b, "spectral_floor_db": f,
                    "reduction": 100.0}, raw_test, out_n[i], voice, out_m[i])
            for i, (b, f) in enumerate(grid)
        ])

    if "neural" in families or "hybrid" in families:
        nn = neural or DeepFilterNetEngine()
        nn.reset()
        warm = learn[-int(0.5 * RATE):]
        n_learn_test = _run(nn, np.concatenate([learn, test]))
        n_learn, n_test = n_learn_test[: len(learn)], n_learn_test[len(learn):]
        nn.reset()
        n_mix = _run(nn, np.concatenate([warm, mix]))[len(warm):]
        done[0] += 1
        emit({"stage": "neural", "progress": round(0.1 + 0.85 * done[0] / total, 3)})
        if "neural" in families:
            new = []
            for r in NEURAL_REDUCTIONS:
                lim = 0.0 if r >= 100 else 1.0 - r / 100.0
                on = n_test + lim * (test - n_test)
                om = n_mix + lim * (mix - n_mix)
                new.append(_score(f"Neural · {r:g} %", "neural", {"engine": "neural", "reduction": r},
                                  raw_test, on, voice, om))
            step("neural", new)
        if "hybrid" in families:
            nl = n_learn[int(0.5 * RATE):] if len(n_learn) > RATE else n_learn
            grid, out_n, out_m, psd_post = spectral_batch(nl, n_test, n_mix, POST_BETAS,
                                                          POST_FLOORS, residual=True)
            step("hybrid", [
                _score(f"Hybrid · {b:g}× · {f:g} dB", "hybrid",
                       {"engine": "hybrid", "post_aggressiveness": b, "post_floor_db": f,
                        "reduction": 100.0}, raw_test, out_n[i], voice, out_m[i])
                for i, (b, f) in enumerate(grid)
            ])
        if psd_post is None:
            psd_post = SpectralEngine(residual=True).psd_of(
                n_learn[int(0.5 * RATE):] if len(n_learn) > RATE else n_learn)

    chosen = select(cands, cfg)
    for c in cands:
        c["chosen"] = c is chosen
    kind = "steady" if stat < 1.5 else ("fluctuating" if stat < 3.5 else "changing")
    label = {"steady": "Steady noise (fan, AC, hum)", "fluctuating": "Mildly fluctuating noise",
             "changing": "Changing noise (voices, traffic, music?)"}[kind]
    emit({"stage": "done", "progress": 1.0})
    return {
        "version": 1,
        "created": time.time(),
        "style": style,
        "kind": kind,
        "label": label,
        "stationarity_db": round(stat, 2),
        "noise_dbfs": round(level_dbfs, 1),
        "tones": tones,
        "rejected_s": round(rejected_s, 2),
        "learned_s": round(len(noise) / RATE, 2),
        "psd_raw": [float(v) for v in psd_raw],
        "psd_post": [float(v) for v in psd_post] if psd_post is not None else None,
        "bands_db": [round(float(v), 2) for v in _mean_band_levels(noise)],
        "display_db": [round(float(v), 1) for v in _display_spectrum(noise)],
        "chosen": _public(chosen),
        "settings": dict(chosen["params"]),
        "candidates": [_public(c) for c in cands],
        "tune_seconds": round(time.perf_counter() - t0, 2),
    }


def select(cands: list[dict], cfg: dict) -> dict:
    best_voice = max(c["voice_sdr"] for c in cands)
    ok = [c for c in cands
          if c["artifact"] <= cfg["artifact_max"] and c["voice_sdr"] >= best_voice - cfg["voice_tol"]]
    for c in cands:
        c["ok"] = c in ok
    if not ok:
        return max(cands, key=lambda c: c["voice_sdr"])
    top = max(c["removal_db"] for c in ok)
    pool = [c for c in ok if c["removal_db"] >= top - 1.0]
    return min(pool, key=lambda c: (c["latency_ms"], -c["removal_db"]))


def _public(c: dict) -> dict:
    return dict(c)


def _mean_band_levels(x: np.ndarray) -> np.ndarray:
    n = 512
    frames = len(x) // n
    p = np.mean([dsp.band_power(x[i * n : (i + 1) * n]) for i in range(frames)], axis=0)
    return 10 * np.log10(p + 1e-20)


def _display_spectrum(x: np.ndarray, n_bands: int = 96) -> np.ndarray:
    f, p = welch(x, RATE, nperseg=4096)
    edges = np.geomspace(30, 20000, n_bands + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        out.append(np.mean(p[m]) if m.any() else np.interp((a + b) / 2, f, p))
    return 10 * np.log10(np.asarray(out) + 1e-20)

"""Offline command-line denoiser: ``python -m anc.denoise noisy.wav clean.wav``.

Runs the exact real-time chain (same engines, mixing and post-processing).
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np


def main() -> None:
    from .config import DEFAULTS, validate
    from .engines import SAMPLE_RATE
    from .pipeline import process_offline

    ap = argparse.ArgumentParser(prog="python -m anc.denoise", description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--engine", default="neural", choices=["neural", "hybrid", "spectral"])
    ap.add_argument("--reduction", type=float, default=100.0, help="0-100 %% of noise removed")
    ap.add_argument("--highpass", type=float, default=80.0, help="high-pass cutoff Hz (0 = off)")
    ap.add_argument("--clarity", type=float, default=0.0, help="voice clarity boost dB (0-9)")
    ap.add_argument("--keep-rate", action="store_true", help="write at the input sample rate")
    args = ap.parse_args()

    import soundfile as sf
    from scipy.signal import resample_poly

    data, rate = sf.read(args.input, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // g, rate // g).astype(np.float32)
    s = validate({**DEFAULTS, "engine": args.engine, "reduction": args.reduction,
                  "highpass_enabled": args.highpass > 0, "highpass_hz": max(20.0, args.highpass),
                  "clarity_db": args.clarity})
    t = time.perf_counter()
    y = process_offline(mono, s)
    dt = time.perf_counter() - t
    out_rate = SAMPLE_RATE
    if args.keep_rate and rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        y = resample_poly(y, rate // g, SAMPLE_RATE // g).astype(np.float32)
        out_rate = rate
    sf.write(args.output, y, out_rate)
    dur = len(mono) / SAMPLE_RATE
    print(f"{args.input} -> {args.output}: {dur:.1f}s audio in {dt:.2f}s ({dur / dt:.0f}x real time)")


if __name__ == "__main__":
    main()

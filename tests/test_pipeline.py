import numpy as np

from anc.config import validate
from anc.engines import create_engine
from anc.pipeline import Pipeline, process_offline


def plain(**kw):
    base = {"highpass_enabled": False, "limiter_enabled": False}
    base.update(kw)
    return validate(base)


def run(pipe, x):
    pad = (-len(x)) % pipe.block
    x = np.concatenate([x, np.zeros(pad + pipe.latency + pipe.block, dtype=np.float32)])
    return np.concatenate([pipe.process(x[k : k + pipe.block]) for k in range(0, len(x), pipe.block)])


def test_zero_reduction_is_transparent(make_voice, make_noise):
    x = make_voice(1.0) + make_noise(1.0)
    pipe = Pipeline(plain(reduction=0), engine=create_engine("spectral"))
    y = run(pipe, x)[pipe.latency : pipe.latency + len(x)]
    np.testing.assert_allclose(y, x, atol=1e-5)


def test_bypass_is_aligned_dry(make_voice):
    x = make_voice(1.0)
    pipe = Pipeline(plain(bypass=True), engine=create_engine("neural"))
    y = run(pipe, x)[pipe.latency : pipe.latency + len(x)]
    np.testing.assert_allclose(y, x, atol=1e-5)


def test_partial_reduction_attenuates_noise_by_expected_amount(make_noise):
    """90 % reduction -> residual noise ~ -20 dB (dry share) once the engine is ~silent."""
    n = make_noise(3.0)
    full = process_offline(n, plain(reduction=100, engine="neural"))
    part = process_offline(n, plain(reduction=90, engine="neural"))
    s = slice(48000, None)
    a_full = 10 * np.log10(np.mean(n[s] ** 2) / np.mean(full[s] ** 2))
    a_part = 10 * np.log10(np.mean(n[s] ** 2) / np.mean(part[s] ** 2))
    assert a_full > a_part
    assert 16 < a_part < 21.5


def test_mute_outputs_silence(make_voice):
    pipe = Pipeline(plain(mute_output=True), engine=create_engine("spectral"))
    y = run(pipe, make_voice(0.5))
    assert not np.any(y)


def test_limiter_holds_ceiling(make_voice):
    x = make_voice(1.0) * 20  # heavily over full scale
    pipe = Pipeline(validate({"limiter_enabled": True, "limiter_ceiling_db": -1.0, "reduction": 0}),
                    engine=create_engine("spectral"))
    y = run(pipe, x)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-6


def test_settings_change_live_without_glitch(make_voice):
    x = make_voice(1.0)
    pipe = Pipeline(plain(), engine=create_engine("spectral"))
    out = []
    for i, k in enumerate(range(0, len(x) - pipe.block, pipe.block)):
        if i == 20:
            pipe.apply_settings(plain(output_gain_db=-12, clarity_db=6, gate_enabled=True, agc_enabled=True))
        out.append(pipe.process(x[k : k + pipe.block]))
    y = np.concatenate(out)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(np.diff(y))) < 0.5  # no hard discontinuities


def test_offline_matches_length(make_voice):
    x = make_voice(1.234)
    y = process_offline(x, plain())
    assert len(y) == len(x) and y.dtype == np.float32


def test_realtime_budget(make_noise):
    """The whole chain must run far faster than real time (block budget 10.7 ms)."""
    import time

    pipe = Pipeline(validate({"engine": "neural", "gate_enabled": True, "agc_enabled": True, "clarity_db": 3}))
    x = make_noise(0.2)[: pipe.block]
    for _ in range(20):
        pipe.process(x)
    t = time.perf_counter()
    n = 100
    for _ in range(n):
        pipe.process(x)
    per_block = (time.perf_counter() - t) / n
    assert per_block < 0.5 * pipe.block / 48000

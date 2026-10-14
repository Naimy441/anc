import numpy as np
import pytest

from anc.engines import create_engine

ENGINES = ["neural", "spectral", "hybrid"]


def run(engine, x, block=2048):
    pad = (-len(x)) % block
    x = np.concatenate([x, np.zeros(pad + 4 * block, dtype=np.float32)])
    return np.concatenate([engine.process(x[k : k + block]) for k in range(0, len(x), block)])


@pytest.mark.parametrize("name", ENGINES)
def test_latency_is_exact(name, make_voice):
    """Cross-correlation peak between clean voice input and output == declared latency."""
    eng = create_engine(name)
    eng.configure({})
    v = make_voice(3.0)
    y = run(eng, v)
    lags = range(0, 4000, 1)
    scores = [float(np.dot(y[l : l + len(v) - 4000], v[: len(v) - 4000])) for l in lags]
    assert int(np.argmax(scores)) == eng.latency


@pytest.mark.parametrize("name", ENGINES)
def test_suppresses_noise(name, make_noise):
    eng = create_engine(name)
    eng.configure({})
    n = make_noise(4.0)
    y = run(eng, n)[eng.latency : eng.latency + len(n)]
    tail = slice(48000, None)  # after adaptation
    atten = 10 * np.log10(np.mean(n[tail] ** 2) / np.mean(y[tail] ** 2))
    assert atten > 15, f"{name} only removed {atten:.1f} dB"


@pytest.mark.parametrize("name", ENGINES)
def test_improves_snr(name, make_voice, make_noise):
    eng = create_engine(name)
    eng.configure({})
    v = make_voice(4.0)
    n = make_noise(4.0, level=0.02)
    y = run(eng, v + n)[eng.latency : eng.latency + len(v)]
    s = slice(48000, None)

    def snr(ref, est):
        a = np.dot(est[s], ref[s]) / np.dot(ref[s], ref[s])
        return 10 * np.log10(np.sum((a * ref[s]) ** 2) / np.sum((est[s] - a * ref[s]) ** 2))

    before, after = snr(v, v + n), snr(v, y)
    assert after > before + 3, f"{name}: {before:.1f} -> {after:.1f} dB"


def test_reset_is_deterministic(make_noise):
    eng = create_engine("neural")
    x = make_noise(0.5)
    a = run(eng, x)
    eng.reset()
    b = run(eng, x)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_spectral_batch_matches_single(make_voice, make_noise):
    """The tuner's parallel candidates run exactly the runtime code path."""
    x = make_voice(1.0) + make_noise(1.0)
    single = create_engine("spectral")
    single.configure({"spectral_aggressiveness": 1.4, "spectral_floor_db": -25})
    batch = create_engine("spectral")
    batch.configure({})
    batch.set_candidates([1.4, 2.0], [-25, -35])
    from anc.autotune import _run

    a = _run(single, x)
    b = _run(batch, x)
    assert b.shape[0] == 2
    np.testing.assert_allclose(a, b[0], atol=1e-6)
    assert not np.allclose(b[0], b[1])


def test_learned_profile_beats_adaptive_on_steady_noise(make_fan, make_voice):
    from anc.autotune import si_sdr

    noise = make_fan(4.0)
    learn, test = noise[: len(noise) // 2], noise[len(noise) // 2 :]
    v = make_voice(len(test) / 48000)
    v = v * (np.std(test) / np.std(v)) * 10 ** (5 / 20)
    mix = test + v
    res = {}
    for mode in ("adaptive", "learned"):
        eng = create_engine("spectral")
        eng.configure({"noise_mode": mode, "spectral_floor_db": -25})
        eng.learned = eng.psd_of(learn)
        y = run(eng, mix)[eng.latency : eng.latency + len(mix)]
        res[mode] = si_sdr(v[20000:], y[20000:])
    assert res["learned"] > res["adaptive"] + 1.0

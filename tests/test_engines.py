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


def test_spectral_learned_profile(make_noise):
    eng = create_engine("spectral")
    eng.configure({"noise_mode": "learned"})
    eng.start_learning(0.5)
    run(eng, make_noise(1.0))
    assert eng.learned is not None and not eng.learning

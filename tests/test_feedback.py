import numpy as np
import pytest
from scipy.signal import iirpeak, lfilter

from anc.config import validate
from anc.feedback import FeedbackGuard
from anc.pipeline import Pipeline


def closed_loop(settings, src, loop_gain=1.6, f_res=5000.0):
    """Speaker -> room -> mic loop: mic = source + resonant(previous output block)."""
    p = Pipeline(validate(settings))
    blk = p.block
    n = len(src) // blk * blk
    b, a = iirpeak(f_res, 8, fs=48000)
    b = b * loop_gain
    zi = np.zeros(2)
    prev = np.zeros(blk, np.float32)
    out = []
    for k in range(0, n, blk):
        fb, zi = lfilter(b, a, prev, zi=zi)
        y = p.process((src[k : k + blk] + fb).astype(np.float32))
        out.append(y)
        prev = y
    return np.concatenate(out), p


def tone_level(y, f):
    b, a = iirpeak(f, 30, fs=48000)
    return 20 * np.log10(np.std(lfilter(b, a, y)) + 1e-12)


@pytest.mark.parametrize("f_res,gain", [(5000.0, 1.6), (2700.0, 3.0), (7500.0, 2.0)])
def test_guard_stops_howling(make_voice, make_noise, f_res, gain):
    src = make_voice(5.0) + make_noise(5.0, 0.002)
    s = {"engine": "spectral", "reduction": 0}
    howl, _ = closed_loop({**s, "feedback_guard": False}, src, gain, f_res)
    safe, p = closed_loop({**s, "feedback_guard": True}, src, gain, f_res)
    tail = slice(-48000 * 2, None)
    assert tone_level(howl[tail], f_res) > -12            # without: a loud howl
    assert tone_level(safe[tail], f_res) < tone_level(howl[tail], f_res) - 35
    assert p.guard.events >= 1
    assert any(abs(n["freq"] - f_res) < 0.08 * f_res for n in p.guard.active())


def test_reacts_before_full_scale(make_voice, make_noise):
    src = make_voice(3.0) + make_noise(3.0, 0.002)
    y, p = closed_loop({"engine": "spectral", "reduction": 0, "feedback_guard": True}, src)
    b, a = iirpeak(5000, 30, fs=48000)
    peak = 20 * np.log10(np.max(np.abs(lfilter(b, a, y))))
    assert peak < -10  # caught while still growing (unguarded it reaches ~ -1 dBFS)
    assert p.guard.last_event["t"] < 0.6


@pytest.mark.parametrize("sens", ["normal", "high"])
def test_no_false_alarms_on_voice_and_music(make_voice, make_noise, sens):
    t = np.arange(48000 * 8) / 48000
    chord = sum(np.sin(2 * np.pi * f0 * h * t) / h for f0 in (261.6, 329.6, 392.0)
                for h in range(1, 8)).astype(np.float32) * 0.05
    for sig in (make_voice(8.0) + make_noise(8.0, 0.01), chord):
        for eng in ("spectral", "neural"):
            p = Pipeline(validate({"engine": eng, "feedback_sensitivity": sens}))
            for k in range(0, len(sig) - p.block, p.block):
                p.process(sig[k : k + p.block])
            assert p.guard.events == 0, (eng, p.guard.active())


def test_notches_deepen_then_release(make_voice, make_noise):
    """A loop strong enough to ring through the first -15 dB cut gets cut deeper;
    once the room goes quiet the notches are released again."""
    src = make_voice(5.0) + make_noise(5.0, 0.002)
    _, p = closed_loop({"engine": "spectral", "reduction": 0}, src, loop_gain=3.0)
    assert min(n["depth"] for n in p.guard.active()) < -15
    g = p.guard
    silence = np.zeros(p.block, np.float32)
    for _ in range(int(60 * 48000 / p.block)):
        g.process(silence)
    assert g.active() == []


def test_agc_frozen_after_feedback(make_voice, make_noise):
    src = make_voice(3.0) + make_noise(3.0, 0.002)
    _, p = closed_loop({"engine": "spectral", "reduction": 0, "feedback_guard": True,
                        "agc_enabled": True}, src)
    assert p.guard.events >= 1
    assert p.agc.gain_db <= 0.5


def test_guard_reset(make_voice, make_noise):
    src = make_voice(3.0) + make_noise(3.0, 0.002)
    _, p = closed_loop({"engine": "spectral", "reduction": 0}, src)
    assert p.guard.active()
    p.request_guard_reset()
    p.process(np.zeros(p.block, np.float32))
    assert p.guard.active() == []


def test_extreme_loop_escalates_then_recovers(make_voice, make_noise):
    """Speaker right next to the mic: narrow cuts can't keep up, so the guard
    lowers the output until the loop is stable, then recovers once it's quiet."""
    src = make_voice(8.0) + make_noise(8.0, 0.002)
    y, p = closed_loop({"engine": "spectral", "reduction": 0}, src, loop_gain=20.0)
    assert p.guard.safety_db <= -6
    assert tone_level(y[-96000:], 5000) < -60
    silence = np.zeros(p.block, np.float32)
    for _ in range(int(30 * 48000 / p.block)):
        p.guard.process(silence)
    assert p.guard.safety_db == 0.0

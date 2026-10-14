import numpy as np
import pytest

from anc import autotune
from anc.config import validate
from anc.pipeline import Pipeline


@pytest.fixture(scope="module")
def fan_profile():
    from conftest import fan_noise

    return autotune.analyze(fan_noise(5.0), style="balanced")


def test_detects_fan_hum(make_fan, make_noise):
    tones = autotune.detect_tones(make_fan(4.0, hum=118.0))
    assert any(abs(t["freq"] - 118) < 4 for t in tones)
    assert autotune.detect_tones(make_noise(4.0)) == []


def test_notches_remove_hum(make_fan):
    from scipy.signal import sosfilt, welch

    x = make_fan(4.0)
    sos = autotune.notch_sos(autotune.detect_tones(x))
    y = sosfilt(sos, x)
    f, px = welch(x[48000:], 48000, nperseg=16384)
    _, py = welch(y[48000:], 48000, nperseg=16384)
    k = np.argmin(np.abs(f - 118))
    assert 10 * np.log10(px[k] / py[k]) > 15  # hum gone
    band = (f > 400) & (f < 1500)
    assert abs(10 * np.log10(px[band].sum() / py[band].sum())) < 0.5  # rest untouched


def test_stationarity_classification(make_fan, make_voice):
    assert autotune.stationarity_db(make_fan(4.0)) < 1.5
    assert autotune.stationarity_db(make_voice(4.0)) > 3.5


def test_speech_during_capture_is_ignored(make_fan, make_voice):
    x = make_fan(5.0)
    x[96000:150000] += make_voice(5.0)[96000:150000]  # ~10 dB above the fan
    kept, cut = autotune.select_noise_only(x)
    assert cut >= 0.6 and len(kept) >= 3.5 * 48000


def test_autotune_finds_clean_strong_setting(fan_profile):
    p = fan_profile
    c = p["chosen"]
    assert p["kind"] == "steady"
    assert len(p["candidates"]) == 42
    assert c["ok"] and c["removal_db"] > 25
    assert c["artifact"] <= autotune.STYLES["balanced"]["artifact_max"]
    best_voice = max(x["voice_sdr"] for x in p["candidates"])
    assert c["voice_sdr"] >= best_voice - autotune.STYLES["balanced"]["voice_tol"]
    # nothing that passed the checks removes clearly more noise
    assert all(x["removal_db"] <= c["removal_db"] + 1.0 for x in p["candidates"] if x["ok"])
    assert validate(p["settings"])["engine"] == c["family"]
    assert len(p["psd_raw"]) == 513 and len(p["psd_post"]) == 513
    assert any(abs(t["freq"] - 118) < 4 for t in p["tones"])


def test_autotune_respects_family_and_style(make_fan):
    p = autotune.analyze(make_fan(4.0), style="gentle", families=["spectral"])
    c = p["chosen"]
    assert c["family"] == "spectral"
    assert c["artifact"] <= autotune.STYLES["gentle"]["artifact_max"]
    assert c["removal_db"] > 12


def test_selection_rules():
    cands = [
        {"label": "a", "removal_db": 40, "artifact": 1.5, "voice_sdr": 10, "latency_ms": 16},
        {"label": "b", "removal_db": 30, "artifact": 0.1, "voice_sdr": 9.8, "latency_ms": 32},
        {"label": "c", "removal_db": 29.5, "artifact": 0.0, "voice_sdr": 9.9, "latency_ms": 16},
        {"label": "d", "removal_db": 35, "artifact": 0.0, "voice_sdr": 5.0, "latency_ms": 16},
    ]
    chosen = autotune.select(cands, autotune.STYLES["balanced"])
    # a: artifacts, d: voice damage -> rejected; b vs c within 1 dB -> lower latency wins
    assert chosen["label"] == "c"
    assert [c["ok"] for c in cands] == [False, True, True, False]


def test_pipeline_applies_profile_and_detects_drift(fan_profile, make_fan):
    s = validate({**fan_profile["settings"], "noise_mode": "learned"})
    pipe = Pipeline(s)
    pipe.set_profile(fan_profile)
    assert pipe.notch.sos is not None
    x = make_fan(6.0, seed=11)
    for k in range(0, len(x) - pipe.block, pipe.block):
        pipe.process(x[k : k + pipe.block])
    assert pipe.quiet_seconds > 3
    assert pipe.m_drift_db < 2.5 and not pipe.stale
    # the room gets 10 dB louder (fan on high) -> stale profile -> re-tune
    louder = make_fan(6.0, seed=12, level_dbfs=-28)
    for k in range(0, len(louder) - pipe.block, pipe.block):
        pipe.process(louder[k : k + pipe.block])
    assert pipe.stale and pipe.m_drift_db > 4
    q = pipe.take_quiet(4.0)
    assert len(q) == 4 * 48000


def test_profile_rescales_with_input_gain(fan_profile):
    pipe = Pipeline(validate({"engine": "spectral", "noise_mode": "learned"}))
    pipe.set_profile({**fan_profile, "input_gain_db": 0.0})
    base = pipe.engine.learned.copy()
    pipe.apply_settings(validate({"engine": "spectral", "noise_mode": "learned", "input_gain_db": 6}))
    np.testing.assert_allclose(pipe.engine.learned, base * 10 ** 0.6, rtol=1e-6)


def test_capture(make_fan):
    pipe = Pipeline(validate({"engine": "spectral"}))
    pipe.start_capture(1.0)
    x = make_fan(1.5)
    for k in range(0, len(x) - pipe.block, pipe.block):
        pipe.process(x[k : k + pipe.block])
    assert pipe.capture_progress == 1.0
    cap = pipe.take_capture()
    assert cap is not None and len(cap) == 48000 and np.std(cap) > 0
    assert pipe.capture_progress is None

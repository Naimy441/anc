import math

import pytest

from anc.config import DEFAULTS, PRESETS, Store, apply_preset, validate
from anc.pipeline import reduction_to_atten_db


def test_defaults_validate_roundtrip():
    assert validate({}) == DEFAULTS


def test_clamping_and_coercion():
    s = validate({"reduction": 250, "input_gain_db": "-99", "gate_enabled": "true", "input_device": "3"})
    assert s["reduction"] == 100.0
    assert s["input_gain_db"] == -24.0
    assert s["gate_enabled"] is True
    assert s["input_device"] == 3


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        validate({"engine": "magic"})
    with pytest.raises(ValueError):
        validate({"reduction": "lots"})
    with pytest.raises(ValueError):
        validate({"reduction": float("nan")})


def test_unknown_keys_ignored():
    assert "evil" not in validate({"evil": 1})


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_apply(name):
    s = apply_preset(name, DEFAULTS)
    assert s["preset"] == name
    assert validate(s) == s


def test_reduction_mapping():
    assert reduction_to_atten_db(0) == 0
    assert math.isclose(reduction_to_atten_db(90), 20.0)
    assert math.isclose(reduction_to_atten_db(99), 40.0)
    assert math.isinf(reduction_to_atten_db(100))


def test_store_persists(tmp_path):
    st = Store(tmp_path / "s.json")
    s = validate({"reduction": 42, "bypass": True})
    st.save(s)
    loaded = st.load()
    assert loaded["reduction"] == 42
    assert loaded["bypass"] is False  # never resume in bypass

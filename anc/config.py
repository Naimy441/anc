"""Settings schema, validation, presets and persistence."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

# key: (type, default, min, max | choices)
SCHEMA: dict[str, tuple] = {
    "engine": (str, "neural", ("neural", "hybrid", "spectral")),
    "reduction": (float, 100.0, 0.0, 100.0),
    "bypass": (bool, False),
    "mute_output": (bool, False),
    "input_gain_db": (float, 0.0, -24.0, 24.0),
    "output_gain_db": (float, 0.0, -24.0, 24.0),
    "highpass_enabled": (bool, True),
    "highpass_hz": (float, 80.0, 20.0, 400.0),
    "clarity_db": (float, 0.0, 0.0, 9.0),
    "gate_enabled": (bool, False),
    "gate_threshold_db": (float, -55.0, -90.0, -20.0),
    "gate_range_db": (float, 30.0, 6.0, 80.0),
    "agc_enabled": (bool, False),
    "agc_target_db": (float, -20.0, -36.0, -10.0),
    "limiter_enabled": (bool, True),
    "limiter_ceiling_db": (float, -1.0, -12.0, 0.0),
    "spectral_aggressiveness": (float, 1.0, 0.5, 3.0),
    "spectral_smoothing": (float, 0.98, 0.8, 0.995),
    "spectral_floor_db": (float, -40.0, -60.0, -6.0),
    "post_aggressiveness": (float, 1.0, 0.5, 3.0),
    "post_floor_db": (float, -18.0, -40.0, -6.0),
    "tune_style": (str, "balanced", ("gentle", "balanced", "max")),
    "auto_retune": (bool, True),
    "hum_notch_enabled": (bool, True),
    "noise_mode": (str, "adaptive", ("adaptive", "learned", "hybrid")),
    "latency_mode": (str, "balanced", ("ultra", "balanced", "safe")),
    "input_device": (object, None),
    "output_device": (object, None),
    "input_channel": (int, -1, -1, 63),
    "preset": (str, "custom", None),
}

DEFAULTS = {k: v[1] for k, v in SCHEMA.items()}

PRESETS: dict[str, dict] = {
    "voice_call": {
        "label": "Voice call",
        "engine": "neural", "reduction": 100.0, "highpass_enabled": True, "highpass_hz": 90.0,
        "clarity_db": 2.0, "gate_enabled": False, "agc_enabled": True, "agc_target_db": -20.0,
        "limiter_enabled": True,
    },
    "studio": {
        "label": "Natural",
        "engine": "neural", "reduction": 85.0, "highpass_enabled": True, "highpass_hz": 60.0,
        "clarity_db": 0.0, "gate_enabled": False, "agc_enabled": False, "limiter_enabled": True,
    },
    "max": {
        "label": "Maximum",
        "engine": "hybrid", "reduction": 100.0, "highpass_enabled": True, "highpass_hz": 100.0,
        "clarity_db": 3.0, "gate_enabled": True, "gate_threshold_db": -50.0, "gate_range_db": 40.0,
        "agc_enabled": True, "limiter_enabled": True, "spectral_aggressiveness": 1.5,
    },
    "low_latency": {
        "label": "Low latency",
        "engine": "spectral", "reduction": 95.0, "highpass_enabled": True, "highpass_hz": 80.0,
        "clarity_db": 1.0, "gate_enabled": False, "agc_enabled": False, "latency_mode": "ultra",
        "spectral_aggressiveness": 1.2, "noise_mode": "adaptive",
    },
}


def validate(updates: dict, base: dict | None = None) -> dict:
    """Return a new settings dict = base + updates, coerced and clamped to the schema.

    Unknown keys are ignored; invalid values raise ``ValueError``.
    """
    out = dict(DEFAULTS if base is None else base)
    for key, value in (updates or {}).items():
        spec = SCHEMA.get(key)
        if spec is None:
            continue
        typ = spec[0]
        if typ is object:
            if value in ("", "default"):
                value = None
            if value is not None and not isinstance(value, (int, str)):
                raise ValueError(f"{key}: invalid device id")
            if isinstance(value, str) and value.lstrip("-").isdigit():
                value = int(value)
            out[key] = value
        elif typ is bool:
            if isinstance(value, str):
                value = value.lower() in ("1", "true", "yes", "on")
            out[key] = bool(value)
        elif typ is str:
            value = str(value)
            choices = spec[2]
            if choices is not None and value not in choices:
                raise ValueError(f"{key}: must be one of {choices}")
            out[key] = value
        else:
            try:
                v = typ(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key}: expected {typ.__name__}") from None
            if isinstance(v, float) and v != v:
                raise ValueError(f"{key}: NaN")
            out[key] = typ(min(spec[3], max(spec[2], v)))
    return out


def apply_preset(name: str, base: dict) -> dict:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}")
    p = {k: v for k, v in PRESETS[name].items() if k != "label"}
    s = validate(p, base)
    s["preset"] = name
    return s


def config_dir() -> Path:
    env = os.environ.get("ANC_CONFIG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".anc"


class Store:
    """Thread-safe persistence: settings in ~/.anc/settings.json, the learned noise
    profile in ~/.anc/profile.json (next to it)."""

    def __init__(self, path: Path | None = None):
        self.path = path or (config_dir() / "settings.json")
        self.profile_path = self.path.with_name("profile.json")
        self.lock = threading.Lock()

    def load_profile(self) -> dict | None:
        try:
            p = json.loads(self.profile_path.read_text())
            return p if isinstance(p, dict) and "psd_raw" in p and "bands_db" in p else None
        except Exception:
            return None

    def save_profile(self, profile: dict | None) -> None:
        with self.lock:
            try:
                if profile is None:
                    self.profile_path.unlink(missing_ok=True)
                    return
                self.profile_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.profile_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(profile))
                tmp.replace(self.profile_path)
            except OSError:
                pass

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text())
            # Never resume in bypass: a fresh start should always be processing.
            data.pop("bypass", None)
            return validate(data)
        except Exception:
            return dict(DEFAULTS)

    def save(self, settings: dict) -> None:
        with self.lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(settings, indent=2))
                tmp.replace(self.path)
            except OSError:
                pass

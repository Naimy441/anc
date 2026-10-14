import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANC_CONFIG_DIR", str(Path(__file__).resolve().parent / ".anc-test"))

RATE = 48000


def voice_like(seconds: float, rate: int = RATE, seed: int = 0) -> np.ndarray:
    """Deterministic harmonic 'voice' with a syllabic envelope (speech stand-in)."""
    t = np.arange(int(seconds * rate)) / rate
    f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    v = sum(np.sin(k * phase) / k for k in range(1, 20))
    env = np.clip(np.sin(2 * np.pi * 1.6 * t), 0, 1) ** 2
    return (0.1 * v * env).astype(np.float32)


def noise(seconds: float, level: float = 0.03, rate: int = RATE, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * rate)) * level).astype(np.float32)


@pytest.fixture
def make_voice():
    return voice_like


@pytest.fixture
def make_noise():
    return noise

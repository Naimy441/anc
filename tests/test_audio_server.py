import io
import time

import numpy as np
import pytest
import soundfile as sf

from anc.audio import AudioWorker
from anc.config import validate
from anc.pipeline import Pipeline


class FakeConn:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


def make_worker(rate_in, rate_out, adaptive, make_voice):
    w = AudioWorker(FakeConn(), simulate=True)
    w.settings = validate({})
    w.pipe = Pipeline(w.settings, engine=w._engine("spectral"))
    w._setup_converters(rate_in, rate_out, adaptive=adaptive)
    return w


@pytest.mark.parametrize("rate", [16000, 22050, 44100, 96000])
def test_resampling_duplex_is_continuous(rate, make_voice):
    """Device at a non-48k rate: in/out rate conversion keeps the stream gap-free."""
    w = make_worker(rate, rate, False, make_voice)
    frames = int(round(w.pipe.block * rate / 48000))
    x = np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate).astype(np.float32) * 0.3
    for k in range(0, len(x) - frames, frames):
        out = np.zeros((frames, 2), np.float32)
        w._duplex_cb(x[k : k + frames, None], out, frames, None, 0)
    assert w.underruns == 0
    assert w.error == ""


def test_split_streams_track_clock_drift(make_voice):
    """Input clock 0.05 % fast vs output: adaptive resampler keeps FIFO bounded."""
    w = make_worker(48000, 48000, True, make_voice)
    blk = w.pipe.block
    x = np.zeros(blk, np.float32)
    produced = consumed = 0.0
    fills = []
    for i in range(6000):  # ~64 s of audio
        produced += blk * 1.0005
        while produced - consumed >= blk:
            w._input_cb(x[:, None], blk, None, 0)
            consumed += blk
        out = np.zeros((blk, 2), np.float32)
        w._output_cb(out, blk, None, 0)
        fills.append(len(w.fifo_out))
    late = fills[-1000:]
    assert max(late) < w.target_fill * 4 + blk * 4
    assert w.error == ""


def test_sim_worker_start_stop(make_voice):
    w = AudioWorker(FakeConn(), simulate=True)
    w.settings = validate({"engine": "spectral"})
    w.start()
    try:
        assert w.running, w.error
        time.sleep(0.5)
        tel = w.telemetry()
        assert tel is not None and len(tel["spec_in"]) == 96
        assert tel["xruns"] == 0
    finally:
        w.stop()
    assert not w.running


def test_server_api(tmp_path, monkeypatch, make_voice, make_noise):
    from fastapi.testclient import TestClient

    from anc.config import Store
    from anc.server import create_app

    app = create_app(simulate=True, store=Store(tmp_path / "s.json"))
    with TestClient(app) as client:
        st = client.get("/api/state").json()
        assert st["settings"]["engine"] == "neural"
        r = client.post("/api/settings", json={"reduction": 75, "engine": "spectral"})
        assert r.status_code == 200 and r.json()["reduction"] == 75
        assert client.post("/api/settings", json={"engine": "nope"}).status_code == 400
        assert client.post("/api/preset/voice_call").json()["preset"] == "voice_call"

        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            ws.send_json({"op": "start"})
            got_tel = False
            for _ in range(200):
                m = ws.receive_json()
                if m["type"] == "telemetry":
                    got_tel = True
                    break
            assert got_tel
            ws.send_json({"op": "stop"})

        buf = io.BytesIO()
        sf.write(buf, make_voice(1.0) + make_noise(1.0), 48000, format="WAV")
        r = client.post("/api/process", files={"file": ("x.wav", buf.getvalue(), "audio/wav")})
        assert r.status_code == 200
        y, sr = sf.read(io.BytesIO(r.content))
        assert sr == 48000 and len(y) == 48000
        bad = client.post("/api/process", files={"file": ("x.wav", b"not audio", "audio/wav")})
        assert bad.status_code == 400


def test_learn_and_tune_flow(tmp_path):
    """Learn & tune end to end: capture from the (simulated) input, tune in the
    server, persist the profile, apply settings, and clear it again."""
    from fastapi.testclient import TestClient

    from anc.config import Store
    from anc.server import create_app

    store = Store(tmp_path / "s.json")
    app = create_app(simulate=True, store=store)
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_json({"op": "learn", "seconds": 2.5})
            stages, profile, settings = [], None, None
            for _ in range(3000):
                m = ws.receive_json()
                if m["type"] == "tune":
                    stages.append(m["data"]["stage"])
                    assert m["data"]["stage"] != "error", m["data"]
                elif m["type"] == "settings":
                    settings = m["data"]
                elif m["type"] == "profile":
                    profile = m["data"]
                    break
        assert profile is not None and "psd_raw" not in profile
        assert stages[0] == "listening" and "done" in stages
        assert settings["noise_mode"] == "learned"
        assert settings["engine"] == profile["chosen"]["family"]
        assert store.load_profile() is not None
        assert client.get("/api/profile").json()["chosen"]["label"] == profile["chosen"]["label"]
        assert client.delete("/api/profile").status_code == 200
        assert client.get("/api/profile").json() is None
        assert store.load_profile() is None
        assert client.get("/api/state").json()["settings"]["noise_mode"] == "adaptive"


def test_worker_requests_background_retune(make_fan):
    """A stale profile + enough noise-only audio -> worker asks the server to re-tune."""
    w = AudioWorker(FakeConn(), simulate=True)
    w.settings = validate({"engine": "spectral", "auto_retune": True})
    w.pipe = Pipeline(w.settings, engine=w._engine("spectral"))
    w.profile = {"bands_db": [0.0] * 24}
    x = make_fan(4.0)
    for k in range(0, len(x) - w.pipe.block, w.pipe.block):
        w.pipe.process(x[k : k + w.pipe.block])
    w.pipe.stale = True
    w._poll_profile_work()
    sent = [m for m in w.conn.sent if m[0] == "captured"]
    assert len(sent) == 1 and sent[0][1]["reason"] == "auto"
    assert len(sent[0][1]["audio"]) >= 3 * 48000
    # cooldown: a second stale flag right away does not re-trigger
    w.pipe.stale = True
    w._poll_profile_work()
    assert len([m for m in w.conn.sent if m[0] == "captured"]) == 1

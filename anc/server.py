"""Local web server: control API, live telemetry websocket and the UI."""
from __future__ import annotations

import asyncio
import io
import json
import math
import multiprocessing as mp
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, autotune
from .audio import worker_main
from .config import DEFAULTS, PRESETS, SCHEMA, Store, apply_preset, validate
from .engines import ENGINES, SAMPLE_RATE

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_FILE_SECONDS = 15 * 60


class Controller:
    """Owns the audio worker process and the authoritative settings."""

    def __init__(self, simulate: bool = False, store: Store | None = None):
        self.simulate = simulate
        self.store = store or Store()
        self.settings = self.store.load()
        self.state: dict = {"running": False, "status": "booting"}
        self.devices: dict = {"inputs": [], "outputs": []}
        self.telemetry: dict | None = None
        self.listeners: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.send_lock = threading.Lock()
        self.want_running = False
        self._closing = False
        self.profile: dict | None = self.store.load_profile()
        self.tuning = False
        self._tune_lock = threading.Lock()
        self._tune_neural = None
        self.last_tune: dict | None = None
        self._spawn()

    # ------------------------------------------------------------ process mgmt
    def _spawn(self) -> None:
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=worker_main, args=(child, self.simulate),
                                daemon=True, name="anc-audio")
        self.proc.start()
        child.close()
        self.send("settings", self.settings)
        self.send("profile", self.profile)
        self.reader = threading.Thread(target=self._read, daemon=True, name="anc-reader")
        self.reader.start()

    def _read(self) -> None:
        conn = self.conn
        while True:
            try:
                kind, payload = conn.recv()
            except (EOFError, OSError):
                break
            if kind == "telemetry":
                self.telemetry = payload
            elif kind == "state":
                self.state = payload
            elif kind == "devices":
                self.devices = payload
            elif kind == "log":
                print(payload, flush=True)
                continue
            elif kind == "captured":
                threading.Thread(target=self._tune, args=(payload,), daemon=True,
                                 name="anc-tune").start()
                continue
            self._emit({"type": kind, **({"data": payload})})
        if self._closing:
            return
        # Worker died (driver crash etc.): restart it and resume if we were running.
        self.state = {"running": False, "status": "error", "error": "Audio engine restarted"}
        self._emit({"type": "state", "data": self.state})
        time.sleep(0.5)
        self._spawn()
        if self.want_running:
            self.send("start", self.settings)

    def _emit(self, msg: dict) -> None:
        loop = self.loop
        if loop is None:
            return
        for q in list(self.listeners):
            loop.call_soon_threadsafe(_offer, q, msg)

    def send(self, cmd: str, payload=None) -> None:
        with self.send_lock:
            try:
                self.conn.send((cmd, payload))
            except (BrokenPipeError, OSError):
                pass

    def close(self) -> None:
        self._closing = True
        self.send("quit")
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.proc.terminate()

    # ---------------------------------------------------------------- actions
    def update(self, values: dict) -> dict:
        new = validate(values, self.settings)
        if "preset" not in values and any(new.get(k) != self.settings.get(k) for k in values
                                          if k not in ("bypass", "mute_output", "input_device",
                                                       "output_device", "input_channel")):
            new["preset"] = "custom"
        return self._commit(new)

    def preset(self, name: str) -> dict:
        return self._commit(apply_preset(name, self.settings))

    def _commit(self, new: dict) -> dict:
        self.settings = new
        self.send("settings", new)
        self.store.save(new)
        self._emit({"type": "settings", "data": new})
        return new

    def start(self) -> None:
        self.want_running = True
        self.send("start", self.settings)

    # ------------------------------------------------------------ learn & tune
    def learn(self, seconds: float = 5.0) -> bool:
        """Start a guided capture of the background noise (auto-starts processing)."""
        if self.tuning:
            return False
        if not self.state.get("running"):
            self.start()
        self.send("capture", max(2.0, min(15.0, float(seconds))))
        self._emit({"type": "tune", "data": {"stage": "listening", "seconds": seconds,
                                             "reason": "user"}})
        return True

    def clear_profile(self) -> None:
        self.profile = None
        self.store.save_profile(None)
        self.send("profile", None)
        self._commit(validate({"noise_mode": "adaptive"}, self.settings))
        self._emit({"type": "profile", "data": None})

    def _tune(self, payload: dict) -> None:
        reason = payload.get("reason", "user")
        if not self._tune_lock.acquire(blocking=False):
            return
        self.tuning = True
        try:
            def progress(m):
                self._emit({"type": "tune", "data": {**m, "reason": reason}})

            progress({"stage": "analyzing", "progress": 0.02})
            if self._tune_neural is None:
                from .engines import DeepFilterNetEngine

                self._tune_neural = DeepFilterNetEngine()
            engine = self.settings.get("engine", "neural")
            # Background re-tunes stay within the running engine (no stream restart).
            families = None if reason == "user" else [engine]
            prof = autotune.analyze(np.asarray(payload["audio"], dtype=np.float32),
                                    style=self.settings.get("tune_style", "balanced"),
                                    families=families, settings=self.settings,
                                    progress=progress, neural=self._tune_neural)
            prof["input_gain_db"] = float(payload.get("input_gain_db", 0.0))
            prof["source"] = reason
            self.profile = prof
            self.store.save_profile(prof)
            self.send("profile", prof)
            upd = dict(prof["settings"])
            if reason != "user":
                upd.pop("engine", None)
            upd["noise_mode"] = "learned"
            new = validate(upd, self.settings)
            new["preset"] = "custom"
            self._commit(new)
            self.last_tune = profile_summary(prof)
            self._emit({"type": "profile", "data": self.last_tune})
        except Exception as e:  # never let a tuning failure take the server down
            self._emit({"type": "tune", "data": {"stage": "error", "error": str(e),
                                                 "reason": reason}})
        finally:
            self.tuning = False
            self._tune_lock.release()

    def stop(self) -> None:
        self.want_running = False
        self.send("stop")

    def snapshot(self) -> dict:
        return {
            "version": __version__,
            "settings": self.settings,
            "state": self.state,
            "devices": self.devices,
            "presets": {k: v["label"] for k, v in PRESETS.items()},
            "engines": ENGINES,
            "profile": profile_summary(self.profile),
            "schema": {k: list(v[1:]) if v[0] is not object else [None] for k, v in SCHEMA.items()},
        }


def profile_summary(p: dict | None) -> dict | None:
    """Profile without the raw PSD arrays (what the UI needs)."""
    if p is None:
        return None
    return {k: v for k, v in p.items() if k not in ("psd_raw", "psd_post")}


class Listener:
    """Per-websocket outbox: control messages are queued, telemetry is lossy
    (only the freshest frame is kept so a slow client never builds up lag)."""

    def __init__(self):
        self.ctrl: deque = deque(maxlen=256)
        self.tel: dict | None = None
        self.event = asyncio.Event()


def _offer(listener: Listener, msg: dict) -> None:
    if msg["type"] == "telemetry":
        listener.tel = msg
    else:
        listener.ctrl.append(msg)
    listener.event.set()


def create_app(simulate: bool = False, store: Store | None = None) -> FastAPI:
    ctl_holder: dict = {}
    offline_engines: dict = {}
    offline_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app):
        c = Controller(simulate=simulate, store=store)
        c.loop = asyncio.get_running_loop()
        ctl_holder["ctl"] = c
        try:
            yield
        finally:
            c.close()

    app = FastAPI(title="ANC · Real-time Noise Cancellation", version=__version__,
                  lifespan=lifespan)

    def ctl() -> Controller:
        return ctl_holder["ctl"]

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/api/state")
    async def get_state():
        snap = ctl().snapshot()
        snap["telemetry"] = ctl().telemetry
        return snap

    @app.post("/api/settings")
    async def post_settings(values: dict):
        try:
            return ctl().update(values)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.post("/api/preset/{name}")
    async def post_preset(name: str):
        try:
            return ctl().preset(name)
        except ValueError as e:
            raise HTTPException(404, str(e)) from None

    @app.post("/api/start")
    async def post_start():
        ctl().start()
        return {"ok": True}

    @app.post("/api/stop")
    async def post_stop():
        ctl().stop()
        return {"ok": True}

    @app.post("/api/learn")
    async def post_learn(seconds: float = 5.0):
        """Learn the background noise for ``seconds`` (stay quiet), then auto-tune."""
        if not ctl().learn(seconds):
            raise HTTPException(409, "A tuning run is already in progress")
        return {"ok": True}

    @app.get("/api/profile")
    async def get_profile():
        return profile_summary(ctl().profile)

    @app.delete("/api/profile")
    async def delete_profile():
        ctl().clear_profile()
        return {"ok": True}

    @app.get("/api/devices")
    async def get_devices(refresh: bool = False):
        if refresh:
            ctl().send("devices")
            await asyncio.sleep(0.3)
        return ctl().devices

    @app.post("/api/process")
    async def process_file(file: UploadFile = File(...)):
        """Offline denoise of an uploaded file through the exact real-time chain."""
        import soundfile as sf
        from scipy.signal import resample_poly

        from .engines import create_engine
        from .pipeline import process_offline

        raw = await file.read()
        try:
            data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception as e:
            raise HTTPException(400, f"Unsupported audio file: {e}") from None
        if len(data) / rate > MAX_FILE_SECONDS:
            raise HTTPException(413, "File too long (max 15 minutes)")
        mono = data.mean(axis=1)
        if rate != SAMPLE_RATE:
            g = math.gcd(int(rate), SAMPLE_RATE)
            x = resample_poly(mono, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
        else:
            x = mono
        s = dict(ctl().settings)
        s.update(bypass=False, mute_output=False)

        def run():
            with offline_lock:
                name = s["engine"]
                eng = offline_engines.get(name)
                if eng is None:
                    eng = offline_engines[name] = create_engine(name)
                eng.reset()
                t0 = time.perf_counter()
                y = process_offline(x, s, engine=eng)
                return y, time.perf_counter() - t0

        y, dt = await asyncio.to_thread(run)
        buf = io.BytesIO()
        sf.write(buf, y.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        dur = len(x) / SAMPLE_RATE
        return Response(
            buf.getvalue(), media_type="audio/wav",
            headers={"X-Duration": f"{dur:.3f}", "X-Process-Seconds": f"{dt:.3f}",
                     "X-Realtime-Factor": f"{(dt / dur) if dur else 0:.4f}",
                     "Content-Disposition": 'attachment; filename="denoised.wav"'},
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        c = ctl()
        q = Listener()
        c.listeners.add(q)
        await websocket.send_text(json.dumps({"type": "hello", "data": c.snapshot()}))

        async def pump():
            while True:
                await q.event.wait()
                q.event.clear()
                while q.ctrl:
                    await websocket.send_text(json.dumps(q.ctrl.popleft(), separators=(",", ":")))
                if q.tel is not None:
                    msg, q.tel = q.tel, None
                    await websocket.send_text(json.dumps(msg, separators=(",", ":")))

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                op = msg.get("op")
                try:
                    if op == "set":
                        c.update(msg.get("values") or {})
                    elif op == "start":
                        c.start()
                    elif op == "stop":
                        c.stop()
                    elif op == "preset":
                        c.preset(str(msg.get("name")))
                    elif op == "learn":
                        c.learn(float(msg.get("seconds", 5.0)))
                    elif op == "clear_profile":
                        c.clear_profile()
                    elif op == "clear_feedback":
                        c.send("clear_feedback")
                    elif op == "devices":
                        c.send("devices")
                    elif op == "reset":
                        c._commit(validate({**DEFAULTS, "input_device": c.settings.get("input_device"),
                                            "output_device": c.settings.get("output_device")}))
                except (ValueError, TypeError) as e:
                    await websocket.send_text(json.dumps({"type": "error", "data": str(e)}))
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            c.listeners.discard(q)
            sender.cancel()

    @app.exception_handler(Exception)
    async def _err(_req, exc):  # pragma: no cover
        return JSONResponse({"detail": str(exc)}, status_code=500)

    return app

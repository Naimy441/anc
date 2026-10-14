"""Audio I/O worker.

Runs in its own process so the PortAudio callback never competes with the web
server for the GIL. The parent talks to it over a ``multiprocessing`` pipe:

    parent -> worker : (cmd, payload)   start / stop / settings / devices / capture / profile
    worker -> parent : (kind, payload)  state / telemetry / devices / captured / error / log

Device handling:
* Full-duplex stream at 48 kHz when both devices support it (zero extra buffering:
  device block == pipeline block).
* Otherwise the device's native rate is used and streaming sinc resamplers
  convert to/from the 48 kHz pipeline.
* If a duplex stream cannot be opened (e.g. two devices with independent clocks
  on WASAPI), separate input/output streams are used with an adaptive
  (variable-ratio) resampler that tracks clock drift between the two devices.
"""
from __future__ import annotations

import math
import os
import threading
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np

from . import dsp
from .engines import SAMPLE_RATE, create_engine
from .pipeline import Pipeline

RATE = SAMPLE_RATE
LATENCY_MODES = {
    # name: (pipeline block multiplier, PortAudio suggested latency in seconds / keyword)
    "ultra": (1, "low"),
    "balanced": (1, 0.03),
    "safe": (2, "high"),
}
RESTART_KEYS = ("engine", "input_device", "output_device", "latency_mode", "input_channel")

N_BANDS = 96
FFT_N = 4096


def _band_edges(n_bands: int = N_BANDS, fmin: float = 30.0, fmax: float = 20000.0):
    freqs = np.fft.rfftfreq(FFT_N, 1.0 / RATE)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    idx = np.searchsorted(freqs, edges)
    # guarantee at least one bin per band
    for i in range(1, len(idx)):
        if idx[i] <= idx[i - 1]:
            idx[i] = idx[i - 1] + 1
    return idx, edges


class SimStream:
    """Stand-in for a PortAudio duplex stream (``--simulate``): feeds a synthetic noisy
    voice-like signal into the callback in real time. Used for tests / demo without
    audio hardware."""

    def __init__(self, samplerate, blocksize, callback, channels=(1, 2), **_):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self.channels = channels
        self.latency = (blocksize / samplerate, blocksize / samplerate)
        self.active = False
        self._t = None
        self._n = 0
        self._rng = np.random.default_rng(1)
        self.last_out = None
        # --simulate-feedback: route the output back into the "mic" through a
        # 4.8 kHz room resonance with loop gain > 1 (a speaker next to a mic).
        self.feedback = os.environ.get("ANC_SIM_FEEDBACK") == "1"
        if self.feedback:
            from scipy.signal import iirpeak

            b, a = iirpeak(4800.0, 8.0, fs=samplerate)
            self._fb = (b * 1.6, a, np.zeros(2))

    def _signal(self, n):
        t = (self._n + np.arange(n)) / self.samplerate
        self._n += n
        # vowel-ish harmonic source with syllabic envelope + pink-ish noise + hum
        f0 = 140 + 25 * np.sin(2 * np.pi * 0.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / self.samplerate
        voice = sum(np.sin(k * phase) / k for k in range(1, 18))
        env = np.clip(np.sin(2 * np.pi * 1.6 * t), 0, 1) ** 2 * (np.sin(2 * np.pi * 0.2 * t) > -0.3)
        noise = np.convolve(self._rng.standard_normal(n + 8), np.ones(8) / 8, "valid")[:n]
        hum = 0.3 * np.sin(2 * np.pi * 60 * t)
        x = 0.08 * voice * env + 0.05 * noise + 0.02 * hum
        return x.astype(np.float32)

    def _run(self):
        period = self.blocksize / self.samplerate
        nxt = time.perf_counter()
        while self.active:
            x = self._signal(self.blocksize)
            if self.feedback and self.last_out is not None:
                from scipy.signal import lfilter

                b, a, zi = self._fb
                fb, zi = lfilter(b, a, self.last_out[:, 0], zi=zi)
                self._fb = (b, a, zi)
                x = (x + fb).astype(np.float32)
            inp = np.repeat(x[:, None], self.channels[0], axis=1)
            out = np.zeros((self.blocksize, self.channels[1]), dtype=np.float32)
            self.callback(inp, out, self.blocksize, None, 0)
            self.last_out = out
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def start(self):
        self.active = True
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self.active = False
        if self._t:
            self._t.join(timeout=1)

    def close(self):
        self.stop()


class AudioWorker:
    def __init__(self, conn: Connection, simulate: bool = False):
        self.conn = conn
        self.simulate = simulate
        self.sd = None
        if not simulate:
            try:
                import sounddevice as sd

                self.sd = sd
            except Exception as e:  # PortAudio missing
                self.send("error", f"Audio backend unavailable: {e}")
        self.settings: dict = {}
        self.engines: dict = {}
        self.pipe: Pipeline | None = None
        self.streams: list = []
        self.running = False
        self.mode = ""
        self.lock = threading.Lock()
        self.xruns = 0
        self.underruns = 0
        self.status = "idle"
        self.error = ""
        self.device_latency = 0.0
        self.rate_in = RATE
        self.rate_out = RATE
        self.in_name = ""
        self.out_name = ""
        self._band_idx, self._band_edges = _band_edges()
        self._win = np.hanning(FFT_N).astype(np.float32)
        self._win_norm = float(np.sum(self._win) / 2)
        self._last_cb = 0.0
        self.profile: dict | None = None
        self.in_native_rate = RATE
        self._last_retune = 0.0

    # ------------------------------------------------------------------ comms
    def send(self, kind: str, payload) -> None:
        try:
            self.conn.send((kind, payload))
        except (BrokenPipeError, EOFError, OSError):
            pass

    def state(self) -> dict:
        return {
            "running": self.running,
            "status": self.status,
            "error": self.error,
            "mode": self.mode,
            "rate_in": self.rate_in,
            "rate_out": self.rate_out,
            "block": self.pipe.block if self.pipe else 0,
            "algo_latency_ms": (self.pipe.latency / RATE * 1000) if self.pipe else 0,
            "device_latency_ms": self.device_latency * 1000,
            "input_name": self.in_name,
            "output_name": self.out_name,
            "input_native_rate": self.in_native_rate,
            "profile_active": self.profile is not None,
            "capturing": bool(self.pipe and self.pipe.capture_progress is not None),
            "simulate": self.simulate,
        }

    # ---------------------------------------------------------------- devices
    def list_devices(self) -> dict:
        if self.simulate or self.sd is None:
            return {
                "inputs": [{"id": "sim", "name": "Simulated noisy voice", "hostapi": "Simulator",
                            "channels": 1, "rate": RATE, "default": True}],
                "outputs": [{"id": "sim", "name": "Null output", "hostapi": "Simulator",
                             "channels": 2, "rate": RATE, "default": True}],
            }
        sd = self.sd
        if not self.running:
            # Re-scan so hot-plugged devices appear.
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        apis = sd.query_hostapis()
        try:
            din, dout = sd.default.device
        except Exception:
            din, dout = -1, -1
        if din is None or din < 0:
            din = apis[0]["default_input_device"] if apis else -1
        if dout is None or dout < 0:
            dout = apis[0]["default_output_device"] if apis else -1
        inputs, outputs = [], []
        for i, d in enumerate(sd.query_devices()):
            api = apis[d["hostapi"]]["name"]
            base = {"id": i, "name": d["name"], "hostapi": api, "rate": d["default_samplerate"]}
            if d["max_input_channels"] > 0:
                inputs.append({**base, "channels": d["max_input_channels"], "default": i == din})
            if d["max_output_channels"] > 0:
                outputs.append({**base, "channels": d["max_output_channels"], "default": i == dout})
        return {"inputs": inputs, "outputs": outputs}

    def _resolve(self, dev, kind: str):
        sd = self.sd
        if dev in (None, "", "default"):
            dev = sd.default.device[0 if kind == "input" else 1]
            if dev is None or dev < 0:
                dev = sd.query_hostapis()[0]["default_" + kind + "_device"]
        dev = int(dev)
        info = sd.query_devices(dev)
        return dev, info

    # ----------------------------------------------------------------- engine
    def _engine(self, name: str):
        eng = self.engines.get(name)
        if eng is None:
            eng = create_engine(name)
            self.engines[name] = eng
        else:
            eng.reset()
        return eng

    # ---------------------------------------------------------------- stream
    def start(self) -> None:
        self.stop(quiet=True)
        s = self.settings
        mult, pa_latency = LATENCY_MODES.get(s.get("latency_mode", "balanced"), (1, 0.03))
        self.status = "starting"
        self.error = ""
        self.send("state", self.state())
        try:
            self.pipe = Pipeline(s, engine=self._engine(s.get("engine", "neural")), block_mult=mult)
            self.pipe.set_profile(self.profile)
            self.xruns = self.underruns = 0
            if self.simulate or self.sd is None:
                if self.sd is None and not self.simulate:
                    raise RuntimeError("PortAudio not available – install it or run with --simulate")
                self._open_sim()
            else:
                self._open_real(pa_latency)
            self.running = True
            self.status = "running"
            self._last_cb = time.perf_counter()
        except Exception as e:
            self.stop(quiet=True)
            self.status = "error"
            self.error = str(e) or e.__class__.__name__
            self.send("log", traceback.format_exc())
        self.send("state", self.state())

    def _open_sim(self) -> None:
        self.rate_in = self.rate_out = RATE
        self.in_name, self.out_name = "Simulated noisy voice", "Null output"
        self.mode = "duplex"
        self._setup_converters(RATE, RATE, adaptive=False)
        st = SimStream(RATE, self.pipe.block, self._duplex_cb, channels=(1, 2))
        self.in_ch_count, self.out_ch_count = 1, 2
        self.streams = [st]
        self.device_latency = sum(st.latency)
        st.start()

    def _extra(self, info, sd):
        api = sd.query_hostapis(info["hostapi"])["name"]
        if "WASAPI" in api:
            try:
                return sd.WasapiSettings(auto_convert=True)
            except TypeError:
                return None
        return None

    def _open_real(self, pa_latency) -> None:
        sd = self.sd
        s = self.settings
        in_dev, in_info = self._resolve(s.get("input_device"), "input")
        out_dev, out_info = self._resolve(s.get("output_device"), "output")
        self.in_name, self.out_name = in_info["name"], out_info["name"]
        self.in_native_rate = int(in_info["default_samplerate"])
        ch_sel = int(s.get("input_channel", -1))
        max_in = int(in_info["max_input_channels"])
        in_ch = min(max_in, max(2, ch_sel + 1) if ch_sel >= 0 else min(max_in, 2))
        in_ch = max(1, in_ch)
        out_ch = max(1, min(2, int(out_info["max_output_channels"])))
        self.in_ch_count, self.out_ch_count = in_ch, out_ch
        latency = pa_latency

        def rate_ok(rate):
            try:
                sd.check_input_settings(device=in_dev, channels=in_ch, dtype="float32",
                                        samplerate=rate, extra_settings=self._extra(in_info, sd))
                sd.check_output_settings(device=out_dev, channels=out_ch, dtype="float32",
                                         samplerate=rate, extra_settings=self._extra(out_info, sd))
                return True
            except Exception:
                return False

        candidates = [RATE]
        for r in (in_info["default_samplerate"], out_info["default_samplerate"], 44100, 96000):
            r = int(r)
            if r not in candidates:
                candidates.append(r)
        errors = []
        for rate in candidates:
            if not rate_ok(rate):
                continue
            blocksize = self.pipe.block if rate == RATE else int(round(self.pipe.block * rate / RATE))
            try:
                self._setup_converters(rate, rate, adaptive=False)
                extra = (self._extra(in_info, sd), self._extra(out_info, sd))
                st = sd.Stream(device=(in_dev, out_dev), samplerate=rate, blocksize=blocksize,
                               channels=(in_ch, out_ch), dtype="float32", latency=latency,
                               callback=self._duplex_cb, extra_settings=extra,
                               prime_output_buffers_using_stream_callback=False)
                st.start()
                self.streams = [st]
                self.rate_in = self.rate_out = rate
                self.mode = "duplex"
                lat = st.latency
                self.device_latency = float(lat[0] + lat[1]) if isinstance(lat, (tuple, list)) else float(lat) * 2
                return
            except Exception as e:
                errors.append(f"duplex@{rate}: {e}")
        # Fall back to independent input/output streams (separate clocks).
        try:
            self._open_split(in_dev, in_info, out_dev, out_info, in_ch, out_ch, latency)
            return
        except Exception as e:
            errors.append(f"split: {e}")
        raise RuntimeError("Could not open audio devices. " + " | ".join(errors[-3:]))

    def _open_split(self, in_dev, in_info, out_dev, out_info, in_ch, out_ch, latency):
        sd = self.sd

        def pick(info, check, dev, ch):
            for r in (RATE, int(info["default_samplerate"]), 44100):
                try:
                    check(device=dev, channels=ch, dtype="float32", samplerate=r,
                          extra_settings=self._extra(info, sd))
                    return r
                except Exception:
                    continue
            raise RuntimeError(f"no usable sample rate for {info['name']}")

        rin = pick(in_info, sd.check_input_settings, in_dev, in_ch)
        rout = pick(out_info, sd.check_output_settings, out_dev, out_ch)
        self._setup_converters(rin, rout, adaptive=True)
        bin_ = int(round(self.pipe.block * rin / RATE))
        bout = int(round(self.pipe.block * rout / RATE))
        ist = sd.InputStream(device=in_dev, samplerate=rin, blocksize=bin_, channels=in_ch,
                             dtype="float32", latency=latency, callback=self._input_cb,
                             extra_settings=self._extra(in_info, sd))
        ost = sd.OutputStream(device=out_dev, samplerate=rout, blocksize=bout, channels=out_ch,
                              dtype="float32", latency=latency, callback=self._output_cb,
                              extra_settings=self._extra(out_info, sd))
        self.streams = [ist, ost]
        self.rate_in, self.rate_out = rin, rout
        self.mode = "split"
        ist.start()
        ost.start()
        self.device_latency = float(ist.latency + ost.latency)

    def _setup_converters(self, rin: int, rout: int, adaptive: bool) -> None:
        blk = self.pipe.block
        self.adaptive = adaptive
        self.rs_in = dsp.StreamResampler(rin, RATE) if rin != RATE else None
        self.rs_out = dsp.StreamResampler(RATE, rout) if (rout != RATE or adaptive) else None
        self.ratio_adj = 0.0
        self.fifo_in = dsp.Fifo(blk * 8)
        self.fifo_out = dsp.Fifo(int(blk * 8 * rout / RATE) + 4096)
        dev_blk = int(math.ceil(blk * rout / RATE))
        # Exact duplex at 48 kHz needs no cushion. Rate conversion / split streams get
        # one device block (+ rounding slop, + another block for drift when split).
        self.direct = (rin == RATE and rout == RATE and not adaptive)
        self.target_fill = 0 if self.direct else dev_blk + 32 + (dev_blk if adaptive else 0)
        if self.target_fill:
            self.fifo_out.push(np.zeros(self.target_fill, dtype=np.float32))
        self.fill_avg = float(self.target_fill)
        self.max_cushion = self.target_fill + 4 * dev_blk

    def _pick_input(self, indata: np.ndarray) -> np.ndarray:
        ch = int(self.settings.get("input_channel", -1))
        if ch >= 0 and ch < indata.shape[1]:
            return indata[:, ch]
        if indata.shape[1] == 1:
            return indata[:, 0]
        return indata.mean(axis=1)

    def _count(self, status) -> None:
        if status:
            self.xruns += 1

    def _produce(self, x: np.ndarray) -> None:
        """Input-rate samples in -> processed output-rate samples queued in fifo_out."""
        x48 = self.rs_in.process(x) if self.rs_in is not None else x
        self.fifo_in.push(x48)
        blk = self.pipe.block
        while len(self.fifo_in) >= blk:
            y = self.pipe.process(self.fifo_in.pop(blk))
            if self.rs_out is not None:
                y = self.rs_out.process(y)
            self.fifo_out.push(y)

    def _consume(self, frames: int) -> np.ndarray:
        if len(self.fifo_out) >= frames:
            return self.fifo_out.pop(frames)
        self.underruns += 1
        have = self.fifo_out.pop(len(self.fifo_out))
        out = np.concatenate([np.zeros(frames - len(have), dtype=np.float32), have])
        # Starved: grow the cushion a little (bounded) so it does not repeat.
        if self.target_fill < self.max_cushion:
            extra = max(32, frames // 4)
            self.target_fill += extra
            self.fifo_out.push(np.zeros(extra, dtype=np.float32))
        return out

    def _duplex_cb(self, indata, outdata, frames, time_info, status) -> None:
        try:
            self._count(status)
            self._last_cb = time.perf_counter()
            x = self._pick_input(indata)
            if self.direct and frames == self.pipe.block:
                y = self.pipe.process(np.ascontiguousarray(x, dtype=np.float32))
            else:
                self._produce(np.ascontiguousarray(x, dtype=np.float32))
                y = self._consume(frames)
            outdata[:] = y[:, None]
        except Exception:
            outdata.fill(0)
            self.error = traceback.format_exc(limit=2)

    def _input_cb(self, indata, frames, time_info, status) -> None:
        try:
            self._count(status)
            self._last_cb = time.perf_counter()
            x = np.ascontiguousarray(self._pick_input(indata), dtype=np.float32)
            with self.lock:
                self._produce(x)
        except Exception:
            self.error = traceback.format_exc(limit=2)

    def _output_cb(self, outdata, frames, time_info, status) -> None:
        try:
            self._count(status)
            with self.lock:
                y = self._consume(frames)
                fill = len(self.fifo_out)
                # Drift compensation: PI-ish control of the output resampler ratio
                # keeps the FIFO around its target fill despite independent clocks.
                self.fill_avg = 0.995 * self.fill_avg + 0.005 * fill
                err = (self.fill_avg - self.target_fill) / max(1, self.target_fill)
                self.ratio_adj = float(np.clip(self.ratio_adj + 2e-6 * err, -0.005, 0.005))
                adj = float(np.clip(0.002 * err, -0.003, 0.003)) + self.ratio_adj
                if self.rs_out is not None:
                    # adj > 0 -> resampler consumes faster -> fewer samples -> fifo drains
                    self.rs_out.set_ratio_adjust(adj)
                if fill > self.target_fill * 4 + frames * 4:
                    self.fifo_out.drop(fill - self.target_fill)  # hard resync
            outdata[:] = y[:, None]
        except Exception:
            outdata.fill(0)
            self.error = traceback.format_exc(limit=2)

    def stop(self, quiet: bool = False) -> None:
        for st in self.streams:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass
        self.streams = []
        self.running = False
        if not quiet:
            self.status = "idle"
            self.send("state", self.state())

    # -------------------------------------------------------------- settings
    def update_settings(self, s: dict) -> None:
        old = self.settings
        self.settings = dict(s)
        if self.pipe is not None:
            if self.running:
                self.pipe.request_settings(self.settings)
            else:
                self.pipe.apply_settings(self.settings)
        if self.running and any(old.get(k) != s.get(k) for k in RESTART_KEYS):
            self.start()
        else:
            self.send("state", self.state())

    # ------------------------------------------------------------- telemetry
    def _spectrum(self, x: np.ndarray) -> list:
        spec = np.fft.rfft(x * self._win)
        p = (spec.real ** 2 + spec.imag ** 2) / (self._win_norm ** 2)
        c = np.cumsum(np.concatenate([[0.0], p]))
        i = self._band_idx
        sums = c[i[1:]] - c[i[:-1]]
        cnt = np.maximum(i[1:] - i[:-1], 1)
        db = 10 * np.log10(sums / cnt + 1e-14)
        return np.round(np.clip(db, -120, 10), 1).tolist()

    def telemetry(self) -> dict | None:
        p = self.pipe
        if not self.running or p is None:
            return None
        inp, out = p.analyzer.latest(FFT_N)
        wi = inp[-1024:].reshape(64, 16)
        wo = out[-1024:].reshape(64, 16)
        block_ms = p.block / RATE * 1000
        stalled = (time.perf_counter() - self._last_cb) > 1.0
        return {
            "t": time.time(),
            "in_db": round(dsp.lin_to_db(p.m_in_rms), 1),
            "out_db": round(dsp.lin_to_db(p.m_out_rms), 1),
            "in_peak_db": round(dsp.lin_to_db(p.m_in_peak), 1),
            "out_peak_db": round(dsp.lin_to_db(p.m_out_peak), 1),
            "removed_db": round(p.m_removed_db, 1),
            "voice": round(p.m_voice, 2),
            "gate_open": bool(p.gate.open) if self.settings.get("gate_enabled") else None,
            "agc_db": round(p.agc.gain_db, 1) if self.settings.get("agc_enabled") else None,
            "limiter_db": round(p.limiter.reduction_db, 1),
            "proc_ms": round(p.m_proc_ms, 3),
            "proc_max_ms": round(p.m_proc_max_ms, 3),
            "cpu": round(100 * p.m_proc_ms / block_ms, 1),
            "xruns": self.xruns,
            "underruns": self.underruns,
            "clips": p.m_clip,
            "stalled": stalled,
            "latency_ms": round(p.latency / RATE * 1000 + self.device_latency * 1000
                                + (self.target_fill / self.rate_out * 1000 if self.target_fill else 0), 1),
            "spec_in": self._spectrum(inp),
            "spec_out": self._spectrum(out),
            "wave_in": np.round(np.abs(wi).max(axis=1), 4).tolist(),
            "wave_out": np.round(np.abs(wo).max(axis=1), 4).tolist(),
            "capture": None if p.capture_progress is None else round(p.capture_progress, 3),
            "profile": self.profile is not None,
            "drift_db": round(p.m_drift_db, 2) if self.profile is not None else None,
            "quiet_s": round(p.quiet_seconds, 1),
            "feedback": {"notches": p.guard.active(), "events": p.guard.events,
                         "last": p.guard.last_event, "safety_db": round(p.guard.safety_db, 1)} if self.settings.get("feedback_guard", True) else None,
        }

    # ------------------------------------------------------------------ loop
    def handle(self, cmd: str, payload) -> bool:
        if cmd == "settings":
            self.update_settings(payload)
        elif cmd == "start":
            if payload:
                self.settings = dict(payload)
            self.start()
        elif cmd == "stop":
            self.stop()
        elif cmd == "devices":
            try:
                self.send("devices", self.list_devices())
            except Exception as e:
                self.send("error", f"Device scan failed: {e}")
        elif cmd == "capture":
            if self.running and self.pipe is not None:
                self.pipe.start_capture(float(payload or 5.0))
            else:
                self.send("error", "Activate processing first, then learn the noise")
                self.send("tune", {"stage": "error", "error": "not running"})
            self.send("state", self.state())
        elif cmd == "clear_feedback":
            if self.pipe is not None:
                self.pipe.request_guard_reset()
        elif cmd == "profile":
            self.profile = payload
            if self.pipe is not None:
                if self.running:
                    self.pipe.request_profile(payload)
                else:
                    self.pipe.set_profile(payload)
            self.send("state", self.state())
        elif cmd == "state":
            self.send("state", self.state())
        elif cmd == "quit":
            self.stop(quiet=True)
            return False
        return True

    def _poll_profile_work(self) -> None:
        """Hand finished captures – and, when the background has drifted, recent
        noise-only audio – to the parent for (re-)tuning."""
        p = self.pipe
        if p is None:
            return
        meta = {"input_gain_db": float(self.settings.get("input_gain_db", 0.0)),
                "engine": self.settings.get("engine", "neural")}
        cap = p.take_capture()
        if cap is not None:
            self.send("captured", {"audio": cap, "reason": "user", **meta})
            self.send("state", self.state())
            return
        now = time.time()
        if (self.profile is not None and p.stale and self.settings.get("auto_retune", True)
                and now - self._last_retune > 45.0 and p.quiet_seconds >= 3.0):
            self._last_retune = now
            p.stale = False
            self.send("captured", {"audio": p.take_quiet(5.0), "reason": "auto", **meta})

    def loop(self) -> None:
        period = 1.0 / 30.0
        nxt = time.perf_counter()
        alive = True
        self.send("devices", self.list_devices())
        self.send("state", self.state())
        while alive:
            timeout = max(0.0, nxt - time.perf_counter())
            try:
                if self.conn.poll(timeout):
                    cmd, payload = self.conn.recv()
                    alive = self.handle(cmd, payload)
                    continue
            except (EOFError, OSError):
                break
            nxt += period
            if time.perf_counter() - nxt > period:
                nxt = time.perf_counter() + period
            tel = self.telemetry()
            if tel is not None:
                self.send("telemetry", tel)
                self._poll_profile_work()
                # Device vanished (unplugged) -> report and stop.
                if self.streams and not self.simulate:
                    try:
                        if not all(st.active for st in self.streams):
                            self.stop(quiet=True)
                            self.status = "error"
                            self.error = "Audio device stopped (unplugged or taken by another app)"
                            self.send("state", self.state())
                    except Exception:
                        pass
        self.stop(quiet=True)


def worker_main(conn: Connection, simulate: bool = False) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        # Best effort: raise priority of the audio process.
        if hasattr(os, "nice"):
            os.nice(-5)
    except Exception:
        pass
    AudioWorker(conn, simulate=simulate).loop()


# ANC studio: real-time neural noise cancellation

ANC studio takes your microphone, removes background noise in real time, and plays
the clean signal to any output device. You control it from an animated dark-mode
web UI on `localhost`.

- **Engine:** **DeepFilterNet3**, a state-of-the-art 48 kHz full-band
  deep-filtering network. It runs as a fully streaming ONNX graph, about 1.1 ms of
  CPU time per 10.7 ms of audio, roughly 10 % of one core.
- **Precise strength control:** the *Noise reduction %* control is defined
  exactly. *X %* removes *X %* of the noise amplitude, so 90 % = 20 dB and
  99 % = 40 dB. Wet and dry signals are mixed sample-aligned, so there is no
  phasing at any setting.
- **Learn & tune:** stay quiet for 5 seconds while it listens to your room (a fan,
  AC or hum). It then tests 42 configurations on noise it didn't train on and
  applies the one that removes the most noise **without adding artifacts or
  damaging your voice**. It also notches out any hum, and re-tunes itself in the
  background if the noise changes. See [Learn & tune](#learn--tune).
- **Three engines:** Neural, Hybrid (neural plus a statistical residual
  suppressor) and Spectral (classic OM-LSA, 16 ms latency, very little CPU).
- **Full voice chain:** high-pass, voice clarity EQ, residual noise gate, AGC,
  output limiter, click-free A/B bypass and mute. Every setting changes live,
  without glitches.
- **Device handling:**
  - Any input and output device, from any host API.
  - Input channel selection.
  - Automatic sample-rate conversion for devices that don't run at 48 kHz.
  - When input and output can't share one stream, separate streams with
    **adaptive clock-drift compensation**.
  - Under-run self-healing, unplug detection and hot-plug rescanning.
- **Fast:** audio runs in its own high-priority process, so the UI and server can
  never stall it. In balanced mode the device block equals the model hop, so
  nothing is added beyond the model latency and the device buffers.
- **Live visuals:** input and output spectrum, a split spectrogram, a
  level-reactive ring, meters, a noise-removed gauge, voice activity, DSP load,
  latency and dropout counters.
- **File lab:** drop in any recording to process it through the same chain, then
  A/B it seamlessly and download the result.

![pipeline](https://img.shields.io/badge/pipeline-48%20kHz%20mono%20float32-6366f1) ![latency](https://img.shields.io/badge/latency-~50%20ms%20neural%20%2F%20~27%20ms%20spectral-22d3ee)

---

## Quick start

**Requirements:** Python 3.10 or newer. On Linux you also need PortAudio (`sudo apt install libportaudio2`).

```bash
# macOS / Linux
./run.sh

# Windows
run.bat
```

The first run creates a `.venv` and installs the dependencies. The UI then opens
at **http://127.0.0.1:8765**. Pick your microphone and output, then press
**Activate** (or the space bar).

Manual alternative:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m anc                 # options: --port 8765 --no-browser --list-devices --simulate
```

> **macOS:** the first time, allow microphone access for your terminal app
> (System Settings → Privacy & Security → Microphone).
>
> **Feedback warning:** if you monitor through speakers while the mic is live,
> you'll get feedback. Use headphones, or press **M** to mute the output.

### Using the clean mic in Zoom, Discord, OBS or Teams

Route ANC's output into a virtual audio cable, then select that cable as the
microphone in the other app:

| OS      | Virtual cable | ANC output → | In Zoom/Discord pick |
|---------|---------------|--------------|----------------------|
| macOS   | [BlackHole 2ch](https://existential.audio/blackhole/) | BlackHole 2ch | BlackHole 2ch |
| Windows | [VB-CABLE](https://vb-audio.com/Cable/) | CABLE Input | CABLE Output |
| Linux   | `pactl load-module module-null-sink sink_name=anc` | anc | Monitor of anc |

---

## Controls

| Control | What it does |
|---|---|
| **Activate** (Space) | Starts or stops real-time processing. |
| **Noise reduction dial** (↑ ↓, scroll, drag) | 0–100 % of the noise amplitude removed. The readout shows the dB equivalent. 100 % means unlimited suppression. |
| **Engine** | **Neural**: DeepFilterNet3, best quality, 32 ms. **Hybrid**: Neural plus an OM-LSA residual suppressor, the deepest suppression, 48 ms. **Spectral**: OM-LSA with MMSE/SPP noise tracking, 16 ms, minimal CPU, best for steady noise. |
| **Bypass** (B) | A/B against the raw input. The comparison is latency-aligned and crossfaded, so it's click-free. |
| **Mute** (M) | Silences the output while processing and metering continue. |
| **Learn & tune** | Listens for 5 s, then finds and applies the best setup for your background. See below. |
| **High-pass** | Removes rumble, hum and handling noise (default 80 Hz, 4th-order Butterworth). |
| **Voice clarity** | Presence peak at 3.2 kHz plus a gentle air shelf. |
| **Residual noise gate** | Expander with hysteresis and hold. It closes on what little is left between words. |
| **Auto gain (AGC)** | Slowly levels your voice to a target loudness. It only adapts while you are speaking. |
| **Output limiter** | Instant attack, smooth release, plus a soft clipper. The output never clips. |
| **Noise profile panel** | The learned spectrum with hum markers, what it removes, live *drift* (how far the current background is from the learned one), tuning style (Gentle / Balanced / Max), auto re-tune, hum notches, and advanced manual tuning. |
| **Buffering** | Ultra-low, Balanced or Safe. Use Safe if you hear crackles. |
| **Presets** | Voice call, Natural, Maximum, Low latency. |

The UI shows live input and output spectra, a split spectrogram (noisy input on
top, cancelled output below), meters with peak hold, noise removed in dB, voice
activity, DSP load, end-to-end latency, and dropout and clipping counters. Settings
persist in `~/.anc/settings.json`.

---

## Learn & tune

This is built for **constant noise**: fans, AC, computer hum, fridges, road rumble
through a window.

1. **Listen (5 s).** Press **Learn & tune** and stay quiet; the fan keeps running.
   Any louder moment, such as a cough, a word or a click, is detected and left
   out, so only the background is learned. Earbud mics work too.
2. **Analyze.**
   - **Hum detection:** finds steady narrow tones such as a fan's blade-pass hum
     or mains hum, with sub-hertz accuracy, and designs notch filters for them.
   - **Steadiness:** measures how steady the noise is and labels it *steady*,
     *mildly fluctuating* or *changing*.
3. **Tune.** The capture is split in half: one half trains the profile, the other
   is unseen test noise. A synthetic test voice is mixed in at 5 dB SNR. Every
   candidate then runs through the **actual runtime engines**:
   - 24 Spectral settings (OM-LSA with your learned profile: over-subtraction × gain floor),
   - 6 Neural strengths (DeepFilterNet3),
   - 12 Hybrid settings (DeepFilterNet3 plus a residual filter trained on what
     the network leaves behind).

   Each candidate is scored on three things:
   - **Noise removed** (dB) on the unseen noise.
   - **Added artifacts:** the log-kurtosis ratio of the leftover noise's
     spectrum, the standard measure of "musical noise". 0 means the leftover is
     just a quieter copy of the original noise.
   - **Voice quality:** SI-SDR of the processed test voice.
4. **Pick and apply.**
   - Candidates with artifacts above the style's limit, or voice quality more
     than the style's tolerance below the best, are rejected.
   - Of the rest, the one that removes the most wins. If two are within 1 dB,
     the lower-latency one wins.
   - The choice is applied immediately. The results screen plots every
     candidate: noise removed against voice quality.
5. **Stay tuned.**
   - The learned profile slowly follows small changes during speech pauses.
   - The background is fingerprinted continuously, in 24 bands, during pauses.
   - If it drifts by more than 4 dB for 2 s (fan switched to high, AC kicks in),
     ANC re-tunes automatically from recent noise-only audio. This stays within
     the current engine, so there's no audio glitch.

| Style | Artifact limit | Voice-quality tolerance | Use when |
|---|---|---|---|
| Gentle | 0.15 | 0.5 dB | Music, podcasts, maximum naturalness |
| Balanced | 0.35 | 1.5 dB | Calls and recordings (default) |
| Max | 0.7 | 3 dB | Very loud fans, where removal matters most |

The profile is saved to `~/.anc/profile.json` and reloaded at the next start. If
you change the input gain, the profile is rescaled automatically.

> **Earbud tip:** Bluetooth earbuds switch to "headset mode" when their mic is
> in use, usually 16 kHz mono. ANC detects this and warns you. It still works
> well for fan noise, but voices sound band-limited, so wired earbuds or a
> separate mic sound better.

## Measured performance

These numbers come from this repo's own benchmarks: 4-core cloud x86 CPU,
single-threaded ONNX Runtime.

| Engine | Algorithmic latency | Full chain CPU per 10.7 ms of audio | Noise removed (real recorded noise) | SI-SDR at 0 dB input SNR |
|---|---|---|---|---|
| Neural | 32 ms | ~1.4 ms (≈13 % of one core) | 27–29 dB | 0.1 → **10.5 dB** |
| Hybrid | 48 ms | ~1.9 ms | 28–32 dB | 0.1 → 8.5 dB |
| Spectral | 16 ms | ~0.5 ms | best on stationary noise (~25 dB in band) | — |

End-to-end latency is the algorithmic latency plus the device buffers. At 48 kHz
in Balanced mode that's about 53 ms for Neural and about 27 ms for Spectral.

---

## Architecture

```
 ┌───────────── browser (localhost) ─────────────┐
 │ index.html / app.css / app.js   (no deps)      │
 │  dial · spectrum · spectrogram · meters · lab  │
 └──────────────▲──────────────┬──────────────────┘
      telemetry 30 Hz (WS)     │ settings / commands (WS + REST)
 ┌──────────────┴──────────────▼──────────────────┐
 │ FastAPI server (anc/server.py)                 │  main process
 │  settings schema + validation + persistence    │
 └──────────────▲──────────────┬──────────────────┘
                │   multiprocessing pipe
 ┌──────────────┴──────────────▼──────────────────┐
 │ Audio worker (anc/audio.py)                    │  separate process, raised priority
 │  PortAudio duplex stream ─► callback:          │
 │   [resample] ► input gain ► high-pass ► ENGINE │
 │   ► reduction mix (latency-aligned dry)        │
 │   ► clarity EQ ► gate ► AGC ► out gain         │
 │   ► limiter ► bypass xfade ► [resample] ► out  │
 └────────────────────────────────────────────────┘
```

| File | Contents |
|---|---|
| `anc/engines.py` | DeepFilterNet3 streaming ONNX (explicit GRU and ring-buffer state per 512-sample hop), the OM-LSA spectral engine, and Hybrid. Each engine has an exact, declared latency. |
| `anc/autotune.py` | Learn & tune: noise-only selection, hum detection and notches, steadiness, the synthetic test voice, artifact (log-kurtosis) and voice (SI-SDR) scoring, and candidate selection. |
| `anc/pipeline.py` | The processing chain, metrics, voice activity detection, profile application, capture, and background-drift monitoring. `process_offline()` runs the identical chain over a file. |
| `anc/dsp.py` | FIFOs, delay lines, IIR filters, gate, AGC, limiter, and a streaming Kaiser-sinc resampler. The resampler measures 90–140 dB SNR, adds 16 samples of delay, and its ratio can vary, which is what the drift control uses. |
| `anc/audio.py` | Device negotiation (48 kHz → native rate → split streams, WASAPI auto-convert), callbacks, drift compensation, telemetry. |
| `anc/server.py`, `anc/config.py` | API, WebSocket, presets, validation. |

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/state` | Settings, engine state, devices and the latest telemetry. |
| `POST /api/settings` | Partial update, e.g. `{"reduction": 85, "engine": "hybrid"}`. |
| `POST /api/start` / `POST /api/stop` | Start or stop processing. |
| `POST /api/preset/{voice_call\|studio\|max\|low_latency}` | Apply a preset. |
| `POST /api/learn?seconds=5` | Learn the background and auto-tune. Progress arrives on the WebSocket as `tune` events, the result as a `profile` event. |
| `GET /api/profile` / `DELETE /api/profile` | Read or clear the learned profile. |
| `GET /api/devices?refresh=1` | List devices, rescanning first if asked. |
| `POST /api/process` | Upload an audio file (multipart field `file`) and get back the denoised WAV. |
| `WS /ws` | Live telemetry and state. Send `{"op": "set", "values": {...}}`, `{"op": "start"}`, and so on. |

### Offline CLI

```bash
python -m anc.denoise noisy.wav clean.wav --engine neural --reduction 95 --clarity 2
```

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                      # 49 tests: engines, latency, mixing math, resampling, clock drift, auto-tune, API
python -m anc --simulate    # runs the whole app with a synthetic noisy-voice source (no audio hardware)
```

### Troubleshooting

- **Crackles or dropouts:** set Buffering to *Safe*, and check the *Dropouts*
  counter in the footer. On Windows, choose the WASAPI versions of your devices
  for the lowest latency.
- **"Could not open audio devices":** another app may hold the device
  exclusively, or the device doesn't support float32. Try the device under a
  different host API in the list.
- **Input clipping toast:** lower *Input gain* or your OS microphone level.
- **No devices on Linux:** install `libportaudio2`, then press the rescan button.

## Credits and licenses

- **DeepFilterNet3:** H. Schröter et al., *"DeepFilterNet: Perceptually Motivated
  Real-Time Speech Enhancement"*, Interspeech 2023. Model weights are
  MIT / Apache-2.0 (see `anc/models/deepfilternet3/`).
- **Streaming ONNX export:** the [torchDF](https://github.com/grazder/DeepFilterNet/tree/torchDF_main/torchDF)
  pure-PyTorch streaming port (hop 512 / FFT 1024 fine-tune).
- **Spectral engine:** T. Gerkmann & R. Hendriks, *Unbiased MMSE-based noise power
  estimation with low complexity* (2012); I. Cohen, *OM-LSA* (2002); Ephraim &
  Malah, *log-spectral amplitude estimator* (1985).

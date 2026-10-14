/* ANC studio – UI controller. No dependencies. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

  // ------------------------------------------------------------------ state
  const S = {
    settings: {}, state: {}, devices: { inputs: [], outputs: [] }, presets: {}, schema: {},
    tel: null, telAt: 0, connected: false, dragging: null, warnedFeedback: false,
  };
  const N = 96;
  const disp = { specIn: new Float32Array(N).fill(-120), specOut: new Float32Array(N).fill(-120),
    ringIn: new Float32Array(64), ringOut: new Float32Array(64),
    inDb: -90, outDb: -90, inPk: -90, outPk: -90, removed: 0, voice: 0, level: 0 };

  // ------------------------------------------------------------- websocket
  let ws, pending = {}, flushQueued = false, retry = 0;
  function connect() {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onopen = () => { S.connected = true; retry = 0; };
    ws.onclose = () => {
      S.connected = false; setStatus("error", "Disconnected · retrying");
      setTimeout(connect, Math.min(4000, 400 * 2 ** retry++));
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      switch (m.type) {
        case "hello":
          Object.assign(S, { settings: m.data.settings, state: m.data.state, devices: m.data.devices,
            presets: m.data.presets, schema: m.data.schema });
          buildPresets(); renderDevices(); renderSettings(); renderState();
          break;
        case "settings": S.settings = m.data; renderSettings(); break;
        case "state": onState(m.data); break;
        case "devices": S.devices = m.data; renderDevices(); break;
        case "telemetry": S.tel = m.data; S.telAt = performance.now(); onTelemetry(m.data); break;
        case "error": toast(m.data, "err"); break;
      }
    };
  }
  function send(op, extra = {}) { if (ws && ws.readyState === 1) ws.send(JSON.stringify({ op, ...extra })); }
  function set(values) {
    Object.assign(pending, values);
    Object.assign(S.settings, values);
    if (!flushQueued) {
      flushQueued = true;
      requestAnimationFrame(() => { flushQueued = false; send("set", { values: pending }); pending = {}; });
    }
  }

  // ---------------------------------------------------------------- status
  function setStatus(cls, text) {
    const p = $("#statusPill");
    p.className = "pill " + cls;
    $("#statusText").textContent = text;
  }
  let lastError = "";
  function onState(st) {
    const was = S.state || {};
    S.state = st;
    if (st.status === "error" && st.error && st.error !== lastError) toast(st.error, "err");
    lastError = st.error || "";
    if (was.learning && !st.learning && st.learned) toast("Noise profile captured – switch Noise model to use it", "info");
    renderState();
  }
  function renderState() {
    const st = S.state || {};
    const running = !!st.running;
    document.body.classList.toggle("live", running);
    const power = $("#power");
    power.classList.toggle("on", running);
    power.classList.toggle("busy", st.status === "starting");
    power.classList.toggle("bypassed", running && !!S.settings.bypass);
    power.setAttribute("aria-pressed", running);
    $("#powerLabel").textContent = st.status === "starting" ? "Starting" : running ? "Active" : "Activate";
    if (!S.connected && st.status !== "running") return;
    if (st.status === "starting") setStatus("starting", "Starting engine…");
    else if (st.status === "error") setStatus("error", "Error");
    else if (running && S.settings.bypass) setStatus("bypass", "Bypassed (A/B)");
    else if (running) setStatus("running", st.simulate ? "Active · simulated" : "Active");
    else setStatus("", "Idle");
    $("#fRate").textContent = running ? (st.rate_in === st.rate_out ? `${st.rate_in / 1000} kHz` : `${st.rate_in / 1000}/${st.rate_out / 1000} kHz`) : "–";
    $("#fBlock").textContent = running ? `${st.block} smp` : "–";
    $("#fMode").textContent = running ? st.mode : "–";
    const learnBtn = $("#learnBtn");
    learnBtn.disabled = !(running && st.supports_learning);
    learnBtn.classList.toggle("on", !!st.learning);
    learnBtn.querySelector("span").textContent = st.learning ? "Listening…" : "Learn noise";
    $("#learnStatus").textContent = st.learning ? "Capturing – stay quiet…" : st.learned ? "Profile captured ✓" : "No profile captured";
    if (!running) { $("#topLatency").textContent = "–"; $("#topCpu").textContent = "–"; }
  }

  // ------------------------------------------------------------- telemetry
  const tBars = $$("#voiceDots i");
  function onTelemetry(t) {
    $("#topLatency").textContent = `${t.latency_ms.toFixed(0)} ms`;
    $("#topCpu").textContent = `${t.cpu.toFixed(0)} %`;
    $("#fProc").textContent = `${t.proc_ms.toFixed(2)} ms`;
    $("#fProcMax").textContent = `${t.proc_max_ms.toFixed(2)} ms`;
    const x = $("#fXruns"); x.textContent = t.xruns + t.underruns; x.classList.toggle("bad", t.xruns + t.underruns > 0);
    const c = $("#fClips"); c.textContent = t.clips; c.classList.toggle("bad", t.clips > 0);
    if (t.clips > 0 && !S.clipWarned) { S.clipWarned = true; toast("Input is clipping – lower the input gain or your mic level", "warn"); }
    pushSpectrogram(t.spec_in, t.spec_out);
  }

  // ------------------------------------------------------------------ dial
  const CX = 160, CY = 160, R = 136, A0 = 135, SWEEP = 270;
  const pt = (a, r = R) => [CX + r * Math.cos(a * Math.PI / 180), CY + r * Math.sin(a * Math.PI / 180)];
  function arcPath(v) {
    v = clamp(v, 0, 1);
    const a1 = A0 + SWEEP * Math.max(v, 0.0001);
    const [x0, y0] = pt(A0), [x1, y1] = pt(a1);
    return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${R} ${R} 0 ${SWEEP * v > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  }
  function buildDial() {
    $("#arcTrack").setAttribute("d", arcPath(1));
    $("#arcHit").setAttribute("d", arcPath(1));
    const g = $("#ticks");
    let html = "";
    for (let i = 0; i <= 20; i++) {
      const a = A0 + SWEEP * i / 20, major = i % 5 === 0;
      const [x0, y0] = pt(a, 150), [x1, y1] = pt(a, major ? 158 : 155);
      html += `<line class="${major ? "major" : ""}" x1="${x0}" y1="${y0}" x2="${x1}" y2="${y1}"/>`;
    }
    g.innerHTML = html;
    const dial = $("#dial");
    const fromEvent = (e) => {
      const r = dial.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width * 320, y = (e.clientY - r.top) / r.height * 320;
      let ang = Math.atan2(y - CY, x - CX) * 180 / Math.PI;
      let rel = (ang - A0 + 720) % 360;
      if (rel > SWEEP) rel = rel > SWEEP + (360 - SWEEP) / 2 ? 0 : SWEEP;
      return Math.round(rel / SWEEP * 100);
    };
    $("#arcHit").addEventListener("pointerdown", (e) => {
      e.preventDefault();
      dial.setPointerCapture(e.pointerId);
      S.dragging = "reduction"; dial.parentElement.classList.add("dragging");
      set({ reduction: fromEvent(e) }); renderReduction();
    });
    dial.addEventListener("pointermove", (e) => { if (S.dragging === "reduction") { set({ reduction: fromEvent(e) }); renderReduction(); } });
    const end = () => { S.dragging = null; dial.parentElement.classList.remove("dragging"); };
    dial.addEventListener("pointerup", end); dial.addEventListener("pointercancel", end);
    dial.addEventListener("keydown", (e) => {
      const d = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 }[e.key];
      if (d) { e.preventDefault(); e.stopPropagation(); bumpReduction(d * (e.shiftKey ? 5 : 1)); }
    });
    dial.addEventListener("wheel", (e) => { e.preventDefault(); bumpReduction(e.deltaY < 0 ? 1 : -1); }, { passive: false });
  }
  function bumpReduction(d) { set({ reduction: clamp(Math.round((S.settings.reduction ?? 100) + d), 0, 100) }); renderReduction(); }
  function atten(p) { return p >= 100 ? Infinity : -20 * Math.log10(1 - p / 100); }
  function renderReduction() {
    const v = S.settings.reduction ?? 100;
    $("#arcValue").setAttribute("d", arcPath(v / 100));
    const [kx, ky] = pt(A0 + SWEEP * v / 100);
    const k = $("#knob"); k.setAttribute("cx", kx); k.setAttribute("cy", ky);
    $("#dial").setAttribute("aria-valuenow", v);
    $("#redValue").textContent = Math.round(v);
    const a = atten(v);
    $("#redSub").textContent = v <= 0 ? "no suppression" : a === Infinity ? "full suppression · max attenuation" : `up to ${a.toFixed(1)} dB attenuation`;
    syncSlider("reduction", v);
  }

  // --------------------------------------------------------------- sliders
  const sliders = {};
  function buildSliders() {
    $$(".slider").forEach((el) => {
      const key = el.dataset.key, unit = el.dataset.unit || "";
      el.innerHTML = `<div class="sl-top"><span>${el.dataset.label}</span><b></b></div>
        <input type="range" min="${el.dataset.min}" max="${el.dataset.max}" step="${el.dataset.step}" aria-label="${el.dataset.label}">`;
      const input = el.querySelector("input"), out = el.querySelector("b");
      const fmt = (v) => {
        const step = parseFloat(el.dataset.step);
        const s = step < 0.01 ? v.toFixed(3) : step < 1 ? v.toFixed(step < 0.1 ? 2 : 1) : v.toFixed(0);
        return (unit === "dB" && v > 0 ? "+" : "") + s + (unit ? " " + unit : "");
      };
      const paint = () => {
        const v = parseFloat(input.value);
        input.style.setProperty("--p", ((v - input.min) / (input.max - input.min) * 100) + "%");
        out.textContent = fmt(v);
      };
      input.addEventListener("input", () => {
        paint(); set({ [key]: parseFloat(input.value) });
        if (key === "reduction") renderReduction();
      });
      input.addEventListener("pointerdown", () => (S.dragging = key));
      input.addEventListener("pointerup", () => (S.dragging = null));
      input.addEventListener("dblclick", () => {
        const def = S.schema[key]?.[0]; if (def != null) { input.value = def; paint(); set({ [key]: def }); if (key === "reduction") renderReduction(); }
      });
      (sliders[key] ||= []).push({ el, input, paint });
    });
  }
  function syncSlider(key, v) {
    (sliders[key] || []).forEach((s) => { if (S.dragging !== key || document.activeElement !== s.input) { s.input.value = v; s.paint(); } });
  }

  // -------------------------------------------------------------- settings
  function renderSettings() {
    const s = S.settings;
    Object.keys(sliders).forEach((k) => s[k] != null && syncSlider(k, s[k]));
    $$("input[type=checkbox][data-key]").forEach((el) => (el.checked = !!s[el.dataset.key]));
    $$("select[data-key]").forEach((el) => {
      if (document.activeElement === el) return;
      const v = s[el.dataset.key];
      el.value = v == null ? "" : String(v);
      if (el.value !== (v == null ? "" : String(v)) && el.options.length) el.value = el.options[0].value;
    });
    $$("#engines button").forEach((b) => b.classList.toggle("on", b.dataset.engine === s.engine));
    $$("#presets button").forEach((b) => b.classList.toggle("on", b.dataset.preset === s.preset));
    $$(".slider[data-dep]").forEach((el) => el.classList.toggle("dim", !s[el.dataset.dep]));
    $("#bypassBtn").classList.toggle("on", !!s.bypass);
    $("#muteBtn").classList.toggle("on", !!s.mute_output);
    $("#spectralGroup").classList.toggle("dim", s.engine === "neural");
    $("#topEngine").textContent = { neural: "Neural", hybrid: "Hybrid", spectral: "Spectral" }[s.engine] || "–";
    renderReduction(); renderState(); renderChannels(); feedbackHint();
  }

  function bindControls() {
    $$("input[type=checkbox][data-key]").forEach((el) => el.addEventListener("change", () => { set({ [el.dataset.key]: el.checked }); renderSettings(); }));
    $$("select[data-key]").forEach((el) => el.addEventListener("change", () => {
      const k = el.dataset.key; let v = el.value;
      if (k === "input_device" || k === "output_device") v = v === "" ? null : parseInt(v, 10);
      if (k === "input_channel") v = parseInt(v, 10);
      set({ [k]: v }); renderSettings();
    }));
    $$("#engines button").forEach((b) => b.addEventListener("click", () => { set({ engine: b.dataset.engine }); renderSettings(); }));
    $("#power").addEventListener("click", togglePower);
    $("#bypassBtn").addEventListener("click", () => { set({ bypass: !S.settings.bypass }); renderSettings(); });
    $("#muteBtn").addEventListener("click", () => { set({ mute_output: !S.settings.mute_output }); renderSettings(); });
    $("#learnBtn").addEventListener("click", () => { send("learn", { seconds: 2.5 }); toast("Stay quiet for 2.5 s while the noise profile is captured", "info"); });
    $("#clearLearned").addEventListener("click", () => send("clear_learned"));
    $("#refreshDevices").addEventListener("click", (e) => { e.preventDefault(); send("devices"); toast("Rescanning audio devices…"); });
    $("#resetBtn").addEventListener("click", () => send("reset"));
    segmented("#specMode", (v) => (viz.specMode = v));
    segmented("#sgramMode", (v) => { viz.sgMode = v; clearSgram(); });
    document.addEventListener("keydown", (e) => {
      if (e.target.matches("input, select, textarea") && e.key !== "Escape") return;
      if (e.code === "Space") { e.preventDefault(); togglePower(); }
      else if (e.key === "b" || e.key === "B") { set({ bypass: !S.settings.bypass }); renderSettings(); }
      else if (e.key === "m" || e.key === "M") { set({ mute_output: !S.settings.mute_output }); renderSettings(); }
      else if (e.key === "ArrowUp" && e.target.id !== "dial") { e.preventDefault(); bumpReduction(e.shiftKey ? 5 : 1); }
      else if (e.key === "ArrowDown" && e.target.id !== "dial") { e.preventDefault(); bumpReduction(e.shiftKey ? -5 : -1); }
    });
  }
  function segmented(sel, cb) {
    $$(sel + " button").forEach((b) => b.addEventListener("click", () => {
      $$(sel + " button").forEach((x) => x.classList.toggle("on", x === b)); cb(b.dataset.v);
    }));
  }
  function togglePower() {
    if (S.state.running || S.state.status === "starting") send("stop");
    else { send("start"); feedbackHint(true); }
  }

  function buildPresets() {
    const box = $("#presets");
    box.innerHTML = "";
    Object.entries(S.presets).forEach(([k, label]) => {
      const b = document.createElement("button");
      b.dataset.preset = k; b.textContent = label;
      b.addEventListener("click", () => send("preset", { name: k }));
      box.appendChild(b);
    });
  }

  // --------------------------------------------------------------- devices
  function fillDeviceSelect(sel, list) {
    const groups = {};
    list.forEach((d) => (groups[d.hostapi] ||= []).push(d));
    const def = list.find((d) => d.default);
    let html = `<option value="">System default${def ? " — " + esc(def.name) : ""}</option>`;
    Object.entries(groups).forEach(([api, ds]) => {
      html += `<optgroup label="${esc(api)}">` + ds.map((d) =>
        `<option value="${d.id}">${esc(d.name)} · ${d.channels}ch · ${(d.rate / 1000).toFixed(1)}k</option>`).join("") + "</optgroup>";
    });
    sel.innerHTML = html;
  }
  function renderDevices() {
    fillDeviceSelect($("#inputDevice"), S.devices.inputs || []);
    fillDeviceSelect($("#outputDevice"), S.devices.outputs || []);
    renderSettings();
  }
  function currentDevice(kind) {
    const list = S.devices[kind === "input" ? "inputs" : "outputs"] || [];
    const id = S.settings[kind + "_device"];
    return list.find((d) => (id == null ? d.default : d.id === id)) || list.find((d) => d.default);
  }
  function renderChannels() {
    const sel = $("#inputChannel");
    const dev = currentDevice("input");
    const n = Math.min(dev ? dev.channels : 2, 32);
    if (sel.options.length !== n + 1) {
      sel.innerHTML = `<option value="-1">Mix all</option>` + Array.from({ length: n }, (_, i) => `<option value="${i}">Channel ${i + 1}</option>`).join("");
    }
    if (document.activeElement !== sel) sel.value = String(S.settings.input_channel ?? -1);
  }
  function feedbackHint(onStart) {
    const out = currentDevice("output");
    const risky = out && /speaker|lautsprecher|haut-parleur|realtek|built-in output|internal/i.test(out.name) && !/headphone|cable|blackhole|virtual|loopback/i.test(out.name);
    $("#deviceHint").classList.toggle("warn", !!risky);
    if (onStart && risky && !S.warnedFeedback && !S.settings.mute_output) {
      S.warnedFeedback = true;
      toast("Output looks like speakers – use headphones to avoid feedback, or press M to mute", "warn");
    }
  }

  // ---------------------------------------------------------------- toasts
  function toast(msg, kind = "info") {
    const el = document.createElement("div");
    el.className = "toast " + kind; el.textContent = String(msg).split("\n")[0].slice(0, 300);
    $("#toasts").appendChild(el);
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 400); }, kind === "err" ? 7000 : 4200);
  }
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ------------------------------------------------------------ visualiser
  const viz = { specMode: "both", sgMode: "split" };
  const FMIN = 30, FMAX = 20000;
  const bandF = Array.from({ length: N }, (_, i) => Math.sqrt(Math.pow(FMAX / FMIN, i / N) * Math.pow(FMAX / FMIN, (i + 1) / N)) * FMIN);
  const fx = (f, w) => Math.log(f / FMIN) / Math.log(FMAX / FMIN) * w;
  const DBMIN = -100, DBMAX = -10;

  function fitCanvas(c) {
    const r = c.getBoundingClientRect(), d = DPR();
    const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; return true; }
    return false;
  }

  const specC = $("#spectrum"), specX = specC.getContext("2d");
  function drawSpectrum() {
    fitCanvas(specC);
    const W = specC.width, H = specC.height, d = DPR(), ctx = specX;
    ctx.clearRect(0, 0, W, H);
    const padB = 20 * d, padL = 34 * d, pw = W - padL - 8 * d, ph = H - padB - 8 * d;
    // grid
    ctx.font = `${10 * d}px ui-monospace, Menlo, monospace`; ctx.textAlign = "center"; ctx.fillStyle = "#4b5168";
    ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.lineWidth = 1;
    [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000].forEach((f) => {
      const x = padL + fx(f, pw); ctx.beginPath(); ctx.moveTo(x, 8 * d); ctx.lineTo(x, 8 * d + ph); ctx.stroke();
      ctx.fillText(f >= 1000 ? f / 1000 + "k" : f, x, H - 6 * d);
    });
    ctx.textAlign = "right";
    for (let db = -90; db <= -20; db += 10) {
      const y = 8 * d + (1 - (db - DBMIN) / (DBMAX - DBMIN)) * ph;
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + pw, y); ctx.stroke();
      ctx.fillText(db, padL - 6 * d, y + 3 * d);
    }
    const ys = (v) => 8 * d + (1 - clamp((v - DBMIN) / (DBMAX - DBMIN), 0, 1)) * ph;
    const path = (arr) => {
      const p = new Path2D();
      p.moveTo(padL, ys(arr[0]));
      for (let i = 0; i < N - 1; i++) {
        const x0 = padL + fx(bandF[i], pw), x1 = padL + fx(bandF[i + 1], pw);
        p.quadraticCurveTo(x0, ys(arr[i]), (x0 + x1) / 2, (ys(arr[i]) + ys(arr[i + 1])) / 2);
      }
      p.lineTo(padL + pw, ys(arr[N - 1]));
      return p;
    };
    const fillUnder = (p) => { const f = new Path2D(p); f.lineTo(padL + pw, 8 * d + ph); f.lineTo(padL, 8 * d + ph); f.closePath(); return f; };
    if (viz.specMode !== "out") {
      const p = path(disp.specIn);
      const g = ctx.createLinearGradient(0, 8 * d, 0, 8 * d + ph);
      g.addColorStop(0, "rgba(251,113,133,0.35)"); g.addColorStop(1, "rgba(251,113,133,0.02)");
      ctx.fillStyle = g; ctx.fill(fillUnder(p));
      ctx.strokeStyle = "rgba(251,113,133,0.85)"; ctx.lineWidth = 1.5 * d; ctx.stroke(p);
    }
    if (viz.specMode !== "in") {
      const p = path(disp.specOut);
      const g = ctx.createLinearGradient(0, 8 * d, 0, 8 * d + ph);
      g.addColorStop(0, "rgba(99,102,241,0.55)"); g.addColorStop(0.6, "rgba(34,211,238,0.18)"); g.addColorStop(1, "rgba(34,211,238,0.0)");
      ctx.fillStyle = g; ctx.fill(fillUnder(p));
      const lg = ctx.createLinearGradient(padL, 0, padL + pw, 0);
      lg.addColorStop(0, "#22d3ee"); lg.addColorStop(0.55, "#818cf8"); lg.addColorStop(1, "#c084fc");
      ctx.save(); ctx.shadowColor = "rgba(34,211,238,0.8)"; ctx.shadowBlur = 12 * d;
      ctx.strokeStyle = lg; ctx.lineWidth = 2 * d; ctx.stroke(p); ctx.restore();
    }
  }

  // spectrogram
  const sgC = $("#sgram"), sgX = sgC.getContext("2d", { willReadFrequently: false });
  const lut = (stops) => {
    const out = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = i / 255; let k = 0;
      while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
      const [t0, c0] = stops[k], [t1, c1] = stops[k + 1], u = (t - t0) / (t1 - t0 || 1);
      for (let j = 0; j < 3; j++) out[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * clamp(u, 0, 1);
    }
    return out;
  };
  const LUT_OUT = lut([[0, [5, 6, 12]], [0.3, [30, 27, 90]], [0.55, [109, 40, 217]], [0.78, [34, 211, 238]], [1, [236, 254, 255]]]);
  const LUT_IN = lut([[0, [5, 6, 12]], [0.3, [70, 14, 40]], [0.55, [190, 18, 60]], [0.8, [251, 113, 133]], [1, [254, 240, 200]]]);
  let sgCol = null;
  function clearSgram() { fitCanvas(sgC); sgX.fillStyle = "#05060c"; sgX.fillRect(0, 0, sgC.width, sgC.height); sgCol = null; }
  function pushSpectrogram(inp, out) {
    if (fitCanvas(sgC)) clearSgram();
    const W = sgC.width, H = sgC.height, d = DPR(), step = Math.max(1, Math.round(2 * d));
    sgX.drawImage(sgC, step, 0, W - step, H, 0, 0, W - step, H);
    if (!sgCol || sgCol.height !== H || sgCol.width !== step) sgCol = sgX.createImageData(step, H);
    const px = sgCol.data;
    const paint = (arr, lutA, y0, h) => {
      for (let y = 0; y < h; y++) {
        const fpos = 1 - y / (h - 1);
        const bi = clamp(Math.floor(fpos * (N - 1)), 0, N - 1);
        const v = clamp(Math.round((arr[bi] - DBMIN) / (DBMAX - DBMIN) * 255), 0, 255);
        for (let x = 0; x < step; x++) {
          const o = ((y0 + y) * step + x) * 4;
          px[o] = lutA[v * 3]; px[o + 1] = lutA[v * 3 + 1]; px[o + 2] = lutA[v * 3 + 2]; px[o + 3] = 255;
        }
      }
    };
    if (viz.sgMode === "split") {
      const h = Math.floor(H / 2);
      paint(inp, LUT_IN, 0, h);
      paint(out, LUT_OUT, h, H - h);
    } else if (viz.sgMode === "in") paint(inp, LUT_IN, 0, H);
    else paint(out, LUT_OUT, 0, H);
    sgX.putImageData(sgCol, W - step, 0);
    if (viz.sgMode === "split") { sgX.fillStyle = "rgba(255,255,255,0.12)"; sgX.fillRect(W - step, Math.floor(H / 2), step, 1); }
  }

  // ring (radial bars around the power orb)
  const ringC = $("#ringCanvas"), ringX = ringC.getContext("2d");
  function drawRing(t) {
    fitCanvas(ringC);
    const W = ringC.width, H = ringC.height, ctx = ringX, cx = W / 2, cy = H / 2, s = W / 320;
    ctx.clearRect(0, 0, W, H);
    const live = !!S.state.running;
    const r0 = 80 * s, span = 42 * s, n = 64;
    for (let pass = 0; pass < 2; pass++) {
      const arr = pass === 0 ? disp.ringIn : disp.ringOut;
      for (let i = 0; i < n; i++) {
        const a = (i / n) * Math.PI * 2 - Math.PI / 2;
        const idle = 0.04 + 0.03 * Math.sin(t / 700 + i * 0.4);
        const v = live ? clamp(arr[i], 0, 1) : idle;
        const len = 3 * s + v * span;
        const x0 = cx + Math.cos(a) * r0, y0 = cy + Math.sin(a) * r0;
        const x1 = cx + Math.cos(a) * (r0 + len), y1 = cy + Math.sin(a) * (r0 + len);
        ctx.lineCap = "round";
        if (pass === 0) { ctx.strokeStyle = `rgba(251,113,133,${live ? 0.28 : 0.08})`; ctx.lineWidth = 4 * s; }
        else {
          const hue = 190 + (i / n) * 90;
          ctx.strokeStyle = `hsla(${hue}, 90%, ${60 + v * 20}%, ${live ? 0.9 : 0.25})`; ctx.lineWidth = 2.4 * s;
        }
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
    }
  }

  // meters
  const dbPct = (db) => clamp((db + 60) / 60, 0, 1) * 100;
  function drawMeters() {
    const m = [["#meterIn", disp.inDb, disp.inPk], ["#meterOut", disp.outDb, disp.outPk]];
    m.forEach(([sel, db, pk]) => {
      const el = $(sel);
      el.querySelector(".lvl").style.width = dbPct(db) + "%";
      el.querySelector(".pk").style.left = `calc(${dbPct(pk)}% - 2px)`;
      el.querySelector(".db").textContent = db < -89 ? "–∞" : db.toFixed(0) + " dB";
    });
    $("#removedVal").textContent = disp.removed.toFixed(1);
    $("#removedBar").style.width = clamp(disp.removed / 40 * 100, 0, 100) + "%";
    const live = !!S.state.running;
    $("#voiceVal").textContent = !live ? "–" : disp.voice > 0.5 ? "Speech" : "Silence";
    tBars.forEach((b, i) => {
      const on = live && disp.voice > (i + 0.5) / tBars.length;
      b.classList.toggle("on", on);
      b.style.height = live ? 3 + (on ? 11 * Math.abs(Math.sin(performance.now() / 160 + i)) * disp.level : 0) + "px" : "3px";
    });
    const pulse = 1 + (live ? disp.level * 0.18 : 0);
    $("#power").style.setProperty("--pulse", pulse.toFixed(3));
  }

  // animation loop: smooth telemetry -> display state
  function frame(t) {
    const tel = S.tel, live = !!S.state.running && tel && performance.now() - S.telAt < 1500;
    const att = 0.55, rel = 0.18;
    const ease = (cur, tgt) => cur + (tgt - cur) * (tgt > cur ? att : rel);
    if (live) {
      for (let i = 0; i < N; i++) { disp.specIn[i] = ease(disp.specIn[i], tel.spec_in[i]); disp.specOut[i] = ease(disp.specOut[i], tel.spec_out[i]); }
      for (let i = 0; i < 64; i++) {
        const j = i < 32 ? i : 63 - i; // mirror for symmetry
        const wi = clamp(Math.sqrt(tel.wave_in[j * 2] || 0) * 1.6, 0, 1), wo = clamp(Math.sqrt(tel.wave_out[j * 2] || 0) * 1.6, 0, 1);
        const bi = Math.floor(j / 32 * 70) + 8;
        const si = clamp((tel.spec_in[bi] + 95) / 75, 0, 1), so = clamp((tel.spec_out[bi] + 95) / 75, 0, 1);
        disp.ringIn[i] = ease(disp.ringIn[i], 0.5 * wi + 0.6 * si);
        disp.ringOut[i] = ease(disp.ringOut[i], 0.5 * wo + 0.6 * so);
      }
      disp.inDb = ease(disp.inDb, tel.in_db); disp.outDb = ease(disp.outDb, tel.out_db);
      disp.inPk = Math.max(tel.in_peak_db, disp.inPk - 0.4); disp.outPk = Math.max(tel.out_peak_db, disp.outPk - 0.4);
      disp.removed = lerp(disp.removed, tel.removed_db, 0.12);
      disp.voice = lerp(disp.voice, tel.voice, 0.3);
      disp.level = lerp(disp.level, clamp((tel.out_db + 60) / 50, 0, 1), 0.25);
    } else {
      for (let i = 0; i < N; i++) { disp.specIn[i] = lerp(disp.specIn[i], -120, 0.08); disp.specOut[i] = lerp(disp.specOut[i], -120, 0.08); }
      for (let i = 0; i < 64; i++) { disp.ringIn[i] *= 0.9; disp.ringOut[i] *= 0.9; }
      disp.inDb = lerp(disp.inDb, -90, 0.1); disp.outDb = lerp(disp.outDb, -90, 0.1);
      disp.inPk = lerp(disp.inPk, -90, 0.1); disp.outPk = lerp(disp.outPk, -90, 0.1);
      disp.removed = lerp(disp.removed, 0, 0.08); disp.voice = lerp(disp.voice, 0, 0.1); disp.level *= 0.9;
    }
    drawSpectrum(); drawRing(t); drawMeters(); labFrame();
    requestAnimationFrame(frame);
  }

  // ------------------------------------------------------------- file lab
  const lab = { ctx: null, orig: null, clean: null, srcs: [], gains: null, playing: false, startAt: 0, offset: 0, which: "clean", url: null };
  function bindLab() {
    const drop = $("#drop"), input = $("#fileInput");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && processFile(e.dataTransfer.files[0]));
    input.addEventListener("change", () => input.files[0] && processFile(input.files[0]));
    $("#labPlay").addEventListener("click", () => (lab.playing ? labStop() : labPlay(lab.offset)));
    segmented("#labAB", (v) => { lab.which = v; labGains(); });
    $("#labWave").addEventListener("click", (e) => {
      if (!lab.clean) return;
      const r = e.currentTarget.getBoundingClientRect();
      const pos = (e.clientX - r.left) / r.width * lab.clean.duration;
      if (lab.playing) labPlay(pos); else { lab.offset = pos; }
    });
  }
  async function processFile(file) {
    const drop = $("#drop");
    drop.classList.add("busy"); $("#dropText").textContent = `Denoising ${file.name}…`;
    labStop();
    try {
      const fd = new FormData(); fd.append("file", file);
      const res = await fetch("/api/process", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      const blob = await res.blob();
      lab.ctx ||= new (window.AudioContext || window.webkitAudioContext)();
      const [o, c] = await Promise.all([
        file.arrayBuffer().then((b) => lab.ctx.decodeAudioData(b)),
        blob.arrayBuffer().then((b) => lab.ctx.decodeAudioData(b)),
      ]);
      lab.orig = o; lab.clean = c; lab.offset = 0;
      if (lab.url) URL.revokeObjectURL(lab.url);
      lab.url = URL.createObjectURL(blob);
      const a = $("#labDownload"); a.href = lab.url; a.download = file.name.replace(/\.[^.]+$/, "") + "_denoised.wav";
      const rtf = parseFloat(res.headers.get("X-Realtime-Factor") || "0");
      $("#labInfo").textContent = `${c.duration.toFixed(1)} s processed in ${(parseFloat(res.headers.get("X-Process-Seconds")) || 0).toFixed(2)} s · ${rtf > 0 ? (1 / rtf).toFixed(0) + "× faster than real time" : ""} · engine: ${S.settings.engine} · reduction ${S.settings.reduction}%`;
      $("#labResult").hidden = false;
      drawLabWave();
      toast("File denoised – press Play and flip Original / Denoised", "info");
    } catch (err) {
      toast("Processing failed: " + err.message, "err");
    } finally {
      drop.classList.remove("busy"); $("#dropText").innerHTML = "Drop another file or <u>browse</u>";
    }
  }
  let labPeaks = null;
  function peaks(buf, n) {
    const ch = buf.getChannelData(0), step = Math.max(1, Math.floor(ch.length / n)), out = new Float32Array(n);
    for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step, e = Math.min(ch.length, j + step); j < e; j += 4) { const v = Math.abs(ch[j]); if (v > m) m = v; } out[i] = m; }
    return out;
  }
  function drawLabWave() {
    const c = $("#labWave"); fitCanvas(c);
    const W = c.width, H = c.height, ctx = c.getContext("2d");
    if (!labPeaks || labPeaks.w !== W) {
      labPeaks = { w: W, o: peaks(lab.orig, W), c: peaks(lab.clean, W) };
      labPeaks.max = labPeaks.o.reduce((a, b) => Math.max(a, b), 1e-4);
    }
    const max = labPeaks.max;
    ctx.clearRect(0, 0, W, H);
    const bar = (arr, col) => { ctx.fillStyle = col; for (let x = 0; x < W; x++) { const h = arr[x] / max * (H / 2 - 4); ctx.fillRect(x, H / 2 - h, 1, h * 2 || 1); } };
    bar(labPeaks.o, "rgba(251,113,133,0.45)");
    const g = ctx.createLinearGradient(0, 0, W, 0); g.addColorStop(0, "#22d3ee"); g.addColorStop(1, "#a78bfa");
    bar(labPeaks.c, g);
  }
  function labGains() {
    if (!lab.gains || !lab.ctx) return;
    const t = lab.ctx.currentTime;
    lab.gains[0].gain.setTargetAtTime(lab.which === "orig" ? 1 : 0, t, 0.012);
    lab.gains[1].gain.setTargetAtTime(lab.which === "clean" ? 1 : 0, t, 0.012);
  }
  function labPlay(offset = 0) {
    if (!lab.clean) return;
    labStop(true);
    lab.ctx.resume();
    if (offset >= lab.clean.duration) offset = 0;
    const ctx = lab.ctx;
    lab.gains = [ctx.createGain(), ctx.createGain()];
    lab.srcs = [lab.orig, lab.clean].map((buf, i) => {
      const s = ctx.createBufferSource(); s.buffer = buf; s.connect(lab.gains[i]).connect(ctx.destination); return s;
    });
    lab.gains[0].gain.value = lab.which === "orig" ? 1 : 0;
    lab.gains[1].gain.value = lab.which === "clean" ? 1 : 0;
    const when = ctx.currentTime + 0.03;
    lab.srcs.forEach((s) => s.start(when, Math.min(offset, s.buffer.duration - 0.01)));
    lab.srcs[1].onended = () => { if (lab.playing && lab.srcs[1] && ctx.currentTime - lab.startAt >= lab.clean.duration - 0.1) labStop(); };
    lab.startAt = when - offset; lab.playing = true;
    $("#labPlay span").textContent = "Pause";
    $("#labPlayIcon").setAttribute("d", "M7 5h4v14H7zM13 5h4v14h-4z");
  }
  function labStop(silent) {
    if (lab.playing && lab.ctx) lab.offset = lab.ctx.currentTime - lab.startAt;
    lab.srcs.forEach((s) => { try { s.onended = null; s.stop(); } catch (_) { /* already stopped */ } });
    lab.srcs = []; lab.playing = false;
    if (!silent) { $("#labPlay span").textContent = "Play"; $("#labPlayIcon").setAttribute("d", "M7 5l12 7-12 7z"); }
    if (lab.clean && lab.offset >= lab.clean.duration - 0.1) lab.offset = 0;
  }
  function labFrame() {
    if (!lab.clean || $("#labResult").hidden) return;
    const c = $("#labWave");
    if (fitCanvas(c)) { labPeaks = null; }
    drawLabWave();
    const pos = lab.playing ? lab.ctx.currentTime - lab.startAt : lab.offset;
    const x = clamp(pos / lab.clean.duration, 0, 1) * c.width;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.fillRect(x, 0, Math.max(1, DPR()), c.height);
  }

  // ------------------------------------------------------------------ boot
  buildDial(); buildSliders(); bindControls(); bindLab(); renderReduction(); clearSgram();
  window.addEventListener("resize", () => clearSgram());
  connect();
  requestAnimationFrame(frame);
})();

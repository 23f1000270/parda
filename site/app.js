// Parda in the browser: detect faces and plates with ONNX Runtime Web, then mask.
// Mirrors the Python package (parda/detect.py, track.py, mask.py) in one fast pass.

const ORT_VERSION = "1.29.0";
const params = new URLSearchParams(location.search);
const ORT_BASE = params.get("ortbase") || `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

const FACE = { url: "models/face-yunet-640.onnx", size: 640, thresh: 0.4, nms: 0.3, margin: 0.18 };
const PLATE = { url: "models/plate-yolov9t-384.onnx", size: 384, thresh: 0.2, nms: 0.5, margin: 0.12 };
const HOLD_FRAMES = 8;
const GROW = 0.06;
const CELLS = 6;
const NOISE = 14;

const $ = (id) => document.getElementById(id);
const view = $("view");
const vctx = view.getContext("2d", { willReadFrequently: true });
const frame = document.createElement("canvas");
const fctx = frame.getContext("2d", { willReadFrequently: true });
const prep = document.createElement("canvas");
const pctx = prep.getContext("2d", { willReadFrequently: true });

let ort = null;
let faceSess = null;
let plateSess = null;
let backend = "";
let stopLoop = null;
let recorder = null;

// ---------- loading ----------

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = res;
    s.onerror = () => rej(new Error(`Could not load ${src}`));
    document.head.appendChild(s);
  });
}

async function createSession(url) {
  const opts = { graphOptimizationLevel: "all" };
  if ("gpu" in navigator) {
    try {
      const s = await ort.InferenceSession.create(url, { ...opts, executionProviders: ["webgpu"] });
      return [s, "WebGPU"];
    } catch (_) { /* fall back to WebAssembly */ }
  }
  const s = await ort.InferenceSession.create(url, { ...opts, executionProviders: ["wasm"] });
  return [s, "WebAssembly"];
}

async function ensureModels() {
  if (faceSess && plateSess) return;
  setStatus("st-backend", "loading models…");
  if (!window.ort) await loadScript(ORT_BASE + "ort.all.min.js");
  ort = window.ort;
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  let b1, b2;
  [faceSess, b1] = await createSession(FACE.url);
  [plateSess, b2] = await createSession(PLATE.url);
  backend = b1 === b2 ? b1 : `${b1} and ${b2}`;
  setStatus("st-backend", `${backend}, on this device`, true);
}

// ---------- preprocessing ----------

function letterboxTensor(src, sw, sh, size, pad, mode) {
  prep.width = size;
  prep.height = size;
  const r = Math.min(size / sh, size / sw);
  const nw = Math.round(sw * r);
  const nh = Math.round(sh * r);
  const left = Math.floor((size - nw) / 2);
  const top = Math.floor((size - nh) / 2);
  pctx.fillStyle = `rgb(${pad},${pad},${pad})`;
  pctx.fillRect(0, 0, size, size);
  pctx.drawImage(src, 0, 0, sw, sh, left, top, nw, nh);
  const px = pctx.getImageData(0, 0, size, size).data;
  const plane = size * size;
  const out = new Float32Array(3 * plane);
  if (mode === "bgr255") {
    for (let i = 0; i < plane; i++) {
      out[i] = px[4 * i + 2];
      out[plane + i] = px[4 * i + 1];
      out[2 * plane + i] = px[4 * i];
    }
  } else {
    for (let i = 0; i < plane; i++) {
      out[i] = px[4 * i] / 255;
      out[plane + i] = px[4 * i + 1] / 255;
      out[2 * plane + i] = px[4 * i + 2] / 255;
    }
  }
  return { tensor: new ort.Tensor("float32", out, [1, 3, size, size]), r, left, top };
}

// ---------- decoding ----------

function nms(dets, iouThresh) {
  dets.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of dets) {
    if (keep.every((k) => iou(k, d) <= iouThresh)) keep.push(d);
  }
  return keep;
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return u > 0 ? inter / u : 0;
}

function unletterbox(d, lb, sw, sh) {
  return {
    ...d,
    x1: Math.max(0, (d.x1 - lb.left) / lb.r),
    y1: Math.max(0, (d.y1 - lb.top) / lb.r),
    x2: Math.min(sw, (d.x2 - lb.left) / lb.r),
    y2: Math.min(sh, (d.y2 - lb.top) / lb.r),
  };
}

async function detectFaces(src, sw, sh) {
  const lb = letterboxTensor(src, sw, sh, FACE.size, 0, "bgr255");
  const out = await faceSess.run({ [faceSess.inputNames[0]]: lb.tensor });
  const dets = [];
  for (const s of [8, 16, 32]) {
    const cols = FACE.size / s;
    const cls = out[`cls_${s}`].data;
    const obj = out[`obj_${s}`].data;
    const bb = out[`bbox_${s}`].data;
    for (let i = 0; i < cls.length; i++) {
      const score = Math.sqrt(Math.min(Math.max(cls[i], 0), 1) * Math.min(Math.max(obj[i], 0), 1));
      if (score < FACE.thresh) continue;
      const r = Math.floor(i / cols);
      const c = i % cols;
      const cx = (c + bb[4 * i]) * s;
      const cy = (r + bb[4 * i + 1]) * s;
      const w = Math.exp(bb[4 * i + 2]) * s;
      const h = Math.exp(bb[4 * i + 3]) * s;
      dets.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score, label: "face" });
    }
  }
  return nms(dets, FACE.nms).map((d) => unletterbox(d, lb, sw, sh));
}

async function detectPlates(src, sw, sh) {
  const lb = letterboxTensor(src, sw, sh, PLATE.size, 114, "rgb01");
  const out = await plateSess.run({ [plateSess.inputNames[0]]: lb.tensor });
  const t = out[plateSess.outputNames[0]];
  const d = t.data;
  const dets = [];
  for (let i = 0; i < t.dims[0]; i++) {
    const score = d[7 * i + 6];
    if (score < PLATE.thresh) continue;
    dets.push({ x1: d[7 * i + 1], y1: d[7 * i + 2], x2: d[7 * i + 3], y2: d[7 * i + 4], score, label: "plate" });
  }
  return nms(dets, PLATE.nms).map((x) => unletterbox(x, lb, sw, sh));
}

// ---------- thorough scan for photos (see parda/detect.py) ----------

function tileOrigins(h, w, size, overlap = 0.15) {
  const step = Math.max(1, Math.floor(size * (1 - overlap)));
  const axis = (n) => {
    if (n <= size) return [0];
    const pos = [];
    for (let p = 0; p <= n - size; p += step) pos.push(p);
    if (pos[pos.length - 1] !== n - size) pos.push(n - size);
    return pos;
  };
  const out = [];
  for (const y of axis(h)) for (const x of axis(w)) out.push([x, y]);
  return out;
}

const tileCanvas = document.createElement("canvas");
const tctx = tileCanvas.getContext("2d", { willReadFrequently: true });

// One letterboxed pass plus native-scale tiles, merged with NMS. Small faces
// in crowds and distant plates survive the second pass.
async function detectTiled(detectFn, cfg, pad, src, sw, sh) {
  let dets = await detectFn(src, sw, sh);
  const size = cfg.size;
  tileCanvas.width = size;
  tileCanvas.height = size;
  for (const [x0, y0] of tileOrigins(sh, sw, size)) {
    const cw = Math.min(size, sw - x0);
    const ch = Math.min(size, sh - y0);
    tctx.fillStyle = `rgb(${pad},${pad},${pad})`;
    tctx.fillRect(0, 0, size, size);
    tctx.drawImage(src, x0, y0, cw, ch, 0, 0, cw, ch);
    const part = await detectFn(tileCanvas, size, size);
    for (const d of part) {
      dets.push({ ...d, x1: d.x1 + x0, y1: d.y1 + y0, x2: Math.min(sw, d.x2 + x0), y2: Math.min(sh, d.y2 + y0) });
    }
  }
  return nms(dets, cfg.nms);
}

// ---------- mask memory (see parda/track.py) ----------

class MaskMemory {
  constructor() { this.tracks = []; }
  reset() { this.tracks = []; }
  update(dets) {
    for (const t of this.tracks) {
      t.box = t.box.map((v, k) => v + t.vel[k % 2]);
    }
    const pairs = [];
    this.tracks.forEach((t, ti) => dets.forEach((d, di) => {
      if (d.label !== t.label) return;
      const v = iou(boxObj(t.box), d);
      if (v >= 0.2) pairs.push([v, ti, di]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const usedT = new Set();
    const usedD = new Set();
    for (const [, ti, di] of pairs) {
      if (usedT.has(ti) || usedD.has(di)) continue;
      usedT.add(ti); usedD.add(di);
      const t = this.tracks[ti];
      const d = dets[di];
      const nb = [d.x1, d.y1, d.x2, d.y2];
      const dc = [(nb[0] + nb[2] - t.box[0] - t.box[2]) / 2, (nb[1] + nb[3] - t.box[1] - t.box[3]) / 2];
      t.vel = [0.5 * t.vel[0] + 0.5 * dc[0], 0.5 * t.vel[1] + 0.5 * dc[1]];
      t.box = nb;
      t.missed = 0;
    }
    this.tracks.forEach((t, ti) => {
      if (usedT.has(ti)) return;
      t.missed += 1;
      const w = t.box[2] - t.box[0];
      const h = t.box[3] - t.box[1];
      const g = GROW / 2;
      t.box = [t.box[0] - w * g, t.box[1] - h * g, t.box[2] + w * g, t.box[3] + h * g];
    });
    dets.forEach((d, di) => {
      if (!usedD.has(di)) this.tracks.push({ box: [d.x1, d.y1, d.x2, d.y2], label: d.label, vel: [0, 0], missed: 0 });
    });
    this.tracks = this.tracks.filter((t) => t.missed <= HOLD_FRAMES);
    return this.tracks.map((t) => ({ ...boxObj(t.box), label: t.label }));
  }
}
const boxObj = (b) => ({ x1: b[0], y1: b[1], x2: b[2], y2: b[3] });
const memory = new MaskMemory();

// ---------- masking (see parda/mask.py) ----------

function expand(b, frac, w, h) {
  const bw = b.x2 - b.x1;
  const bh = b.y2 - b.y1;
  return {
    x1: Math.max(0, Math.floor(b.x1 - bw * frac)),
    y1: Math.max(0, Math.floor(b.y1 - bh * frac)),
    x2: Math.min(w, Math.ceil(b.x2 + bw * frac)),
    y2: Math.min(h, Math.ceil(b.y2 + bh * frac)),
  };
}

function maskRegion(ctx, b) {
  const w = b.x2 - b.x1;
  const h = b.y2 - b.y1;
  if (w < 2 || h < 2) return;
  const img = ctx.getImageData(b.x1, b.y1, w, h);
  const px = img.data;
  const cell = Math.max(1, Math.ceil(Math.max(w, h) / CELLS));
  for (let cy = 0; cy < h; cy += cell) {
    for (let cx = 0; cx < w; cx += cell) {
      let r = 0, g = 0, bl = 0, n = 0;
      const ey = Math.min(h, cy + cell);
      const ex = Math.min(w, cx + cell);
      for (let y = cy; y < ey; y++) {
        for (let x = cx; x < ex; x++) {
          const i = 4 * (y * w + x);
          r += px[i]; g += px[i + 1]; bl += px[i + 2]; n++;
        }
      }
      const j = Math.round((Math.random() * 2 - 1) * NOISE);
      ctx.fillStyle = `rgb(${clamp(r / n + j)},${clamp(g / n + j)},${clamp(bl / n + j)})`;
      ctx.fillRect(b.x1 + cx, b.y1 + cy, ex - cx, ey - cy);
    }
  }
}
const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));

function failClosed(w, h, msg) {
  vctx.fillStyle = "#17153A";
  vctx.fillRect(0, 0, w, h);
  vctx.fillStyle = "#F3F2EC";
  vctx.font = "600 28px Mukta, sans-serif";
  vctx.fillText(msg, 32, h / 2);
}

// ---------- one frame ----------

async function processFrame(source, sw, sh, useMemory, thorough = false) {
  // Freeze the exact frame we analyse, so what we show is what we checked.
  if (frame.width !== sw || frame.height !== sh) { frame.width = sw; frame.height = sh; }
  fctx.drawImage(source, 0, 0, sw, sh);
  const t0 = performance.now();
  let dets;
  try {
    if (thorough) {
      dets = [
        ...(await detectTiled(detectFaces, FACE, 0, frame, sw, sh)),
        ...(await detectTiled(detectPlates, PLATE, 114, frame, sw, sh)),
      ];
    } else {
      dets = [...(await detectFaces(frame, sw, sh)), ...(await detectPlates(frame, sw, sh))];
    }
  } catch (e) {
    if (view.width !== sw || view.height !== sh) { view.width = sw; view.height = sh; }
    failClosed(sw, sh, "Detection failed, so this frame is hidden.");
    console.error(e);
    return null;
  }
  const boxes = useMemory ? memory.update(dets) : dets;
  const masks = boxes.map((b) => expand(b, b.label === "face" ? FACE.margin : PLATE.margin, sw, sh));
  if (view.width !== sw || view.height !== sh) { view.width = sw; view.height = sh; }
  vctx.drawImage(frame, 0, 0);
  masks.forEach((m) => maskRegion(vctx, m));
  if ($("outline").checked) {
    vctx.strokeStyle = "#F2C318";
    vctx.lineWidth = Math.max(2, Math.round(sw / 400));
    masks.forEach((m) => vctx.strokeRect(m.x1, m.y1, m.x2 - m.x1, m.y2 - m.y1));
  }
  const faces = dets.filter((d) => d.label === "face").length;
  const plates = dets.length - faces;
  setStatus("st-count", `${faces} ${faces === 1 ? "face" : "faces"}, ${plates} ${plates === 1 ? "plate" : "plates"}`);
  return performance.now() - t0;
}

// ---------- UI ----------

function setStatus(id, text, ok = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("ok", ok);
}

function stopAll() {
  if (stopLoop) { stopLoop(); stopLoop = null; }
  if (recorder && recorder.state !== "inactive") recorder.stop();
  memory.reset();
  $("save").hidden = true;
}

function showStage() {
  $("empty").hidden = true;
  view.parentElement.classList.add("live");
}

function fpsMeter() {
  let last = performance.now();
  let avg = 0;
  return () => {
    const now = performance.now();
    const f = 1000 / (now - last);
    last = now;
    avg = avg ? 0.85 * avg + 0.15 * f : f;
    setStatus("st-fps", `${avg.toFixed(1)} frames per second`);
  };
}

async function startCamera() {
  stopAll();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) {
    setStatus("st-backend", "Camera blocked. Allow camera access for this site, then try again.");
    return;
  }
  await ensureModels();
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();
  showStage();
  let running = true;
  const tick = fpsMeter();
  stopLoop = () => { running = false; stream.getTracks().forEach((t) => t.stop()); };
  while (running) {
    if (video.videoWidth) {
      await processFrame(video, video.videoWidth, video.videoHeight, true);
      tick();
    }
    await new Promise(requestAnimationFrame);
  }
}

async function openPhoto(file) {
  stopAll();
  await ensureModels();
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, 1920 / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * scale);
  const h = Math.round(bmp.height * scale);
  showStage();
  setStatus("st-fps", "scanning the whole photo…");
  const ms = await processFrame(bmp, w, h, false, true);
  if (ms != null) setStatus("st-fps", `${Math.round(ms)} ms for this photo`);
  view.toBlob((blob) => offerSave(blob, file.name.replace(/\.[^.]+$/, "") + "_parda.png", "Save blurred photo"), "image/png");
}

async function openVideo(file) {
  stopAll();
  await ensureModels();
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.src = URL.createObjectURL(file);
  await new Promise((r) => { video.onloadedmetadata = r; });
  showStage();
  const tick = fpsMeter();
  // Record only what is drawn on the canvas: masked frames, no audio track.
  const chunks = [];
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
  recorder = new MediaRecorder(view.captureStream(30), { mimeType: mime });
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => offerSave(new Blob(chunks, { type: "video/webm" }), file.name.replace(/\.[^.]+$/, "") + "_parda.webm", "Save blurred video");
  let running = true;
  stopLoop = () => { running = false; video.pause(); };
  // Prime the canvas before recording starts, so the first recorded frame is masked.
  await processFrame(video, video.videoWidth, video.videoHeight, true);
  recorder.start(500);
  await video.play();
  while (running && !video.ended) {
    await processFrame(video, video.videoWidth, video.videoHeight, true);
    tick();
    await new Promise(requestAnimationFrame);
  }
  if (recorder.state !== "inactive") recorder.stop();
}

function offerSave(blob, name, label) {
  const a = $("save");
  if (a.href) URL.revokeObjectURL(a.href);
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.textContent = label;
  a.hidden = false;
}

$("btn-camera").addEventListener("click", () => startCamera().catch(showError));
$("in-photo").addEventListener("change", (e) => e.target.files[0] && openPhoto(e.target.files[0]).catch(showError));
$("in-video").addEventListener("change", (e) => e.target.files[0] && openVideo(e.target.files[0]).catch(showError));

function showError(e) {
  console.error(e);
  setStatus("st-backend", `Stopped: ${e.message}`);
}

// Expose a hook for automated tests.
window.parda = { ensureModels, processFrame, openPhoto };

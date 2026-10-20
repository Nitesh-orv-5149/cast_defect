// Reference-images tab: is the likely type right? Uses the anomaly maps from types_ref_maps.mjs and the same photos
// resized to the app's 448 px. usage: DIR=photos_station MAPS=station_maps.json [FAM=bi] [GT=station_gt.json] node types_ref.js
const fs = require("fs"), jpeg = require("jpeg-js"), CAD = require("../../../web/cad.js");
const D = __dirname + "/../data/" + (process.env.DIR || "photos_station") + "/", maps = JSON.parse(fs.readFileSync(__dirname + "/" + (process.env.MAPS || "station_maps.json")));
const gt = process.env.GT ? JSON.parse(fs.readFileSync(__dirname + "/" + process.env.GT)) : {};
const EXPECT = { crack: /crack/, blindhole: /pit|porosity|surface defect/, throughhole: /blowhole \(through\)/, chip: /chipped|broken or unfilled edge/, brokenblade: /missing or deformed|broken/,
  blocked: /blocked/, throughcrack: /through crack|crack/, flash: /flash/, misrun: /misrun|chipped|broken/, shrinkage: /through|porosity|pit/ };
function resize(src, sw, sh, n) {
  const out = new Uint8ClampedArray(n * n * 4), s = Math.min(sw, sh), ox = (sw - s) / 2, oy = (sh - s) / 2, f = s / n;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = ox + (x + 0.5) * f - 0.5, v = oy + (y + 0.5) * f - 0.5, x0 = Math.max(0, Math.floor(u)), y0 = Math.max(0, Math.floor(v)), x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1), a = u - x0, b = v - y0;
    for (let c = 0; c < 4; c++) out[(y * n + x) * 4 + c] = (1 - a) * (1 - b) * src[(y0 * sw + x0) * 4 + c] + a * (1 - b) * src[(y0 * sw + x1) * 4 + c] + (1 - a) * b * src[(y1 * sw + x0) * 4 + c] + a * b * src[(y1 * sw + x1) * 4 + c];
  }
  return out;
}
const load = id => { const img = jpeg.decode(fs.readFileSync(D + id + ".jpg"), { useTArray: true, formatAsRGBA: true }); return resize(img.data, img.width, img.height, 448); };
const refs = JSON.parse(fs.readFileSync(D + "labels.json")).filter(r => r.kind === "good" && (!process.env.FAM || r.family === process.env.FAM)).slice(0, +(process.env.NREF || 6)).map(r => CAD.surfaceRef(load(r.id), 448, 448));
const g = {}; let ms = 0;
for (const r of maps) {
  const img = jpeg.decode(fs.readFileSync(D + r.id + ".jpg"), { useTArray: true, formatAsRGBA: true }), px = resize(img.data, img.width, img.height, 448);
  const t = Date.now(), f = CAD.typeSurface(px, 448, 448, Float32Array.from(r.d), 28, 28, r.thr, refs); ms += Date.now() - t;
  const top = f[0], cell = top ? Math.floor(top.cy * 28) * 28 + Math.floor(top.cx * 28) : -1;
  const onDef = gt[r.id] ? [-29, -28, -27, -1, 0, 1, 27, 28, 29].some(o => gt[r.id].cells[cell + o]) : null;
  const ok = r.kind === "good" ? null : EXPECT[r.what].test(top ? top.defect : "");
  (g[r.what] ||= []).push({ ok, onDef, v: r.verdict, t: f.map(x => x.defect).join(" + ") });
}
let T = [0, 0], TL = [0, 0];
for (const [k, v] of Object.entries(g)) {
  const n = v.filter(x => x.ok).length, nl = v.filter(x => x.onDef).length;
  if (k !== "good") { T[0] += n; T[1] += v.length; const L = v.filter(x => x.onDef); TL[0] += L.filter(x => x.ok).length; TL[1] += L.length; }
  console.log(`${k.padEnd(13)} type ok ${k === "good" ? "-" : n + "/" + v.length}  located ${nl}/${v.length}  ${v.map(x => (x.ok ? "✓" : x.ok === false ? "✗" : " ") + x.t).join(" | ")}`);
}
console.log(`\nprimary type right: ${T[0]}/${T[1]} | when the hot area is on the defect: ${TL[0]}/${TL[1]} | ${(ms / maps.length).toFixed(0)} ms/photo`);

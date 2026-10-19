// Render the M0 photo set with photo.html in headless Chromium (software WebGL).
// usage: node photo/render_photos.mjs <playwright node_modules dir>   (run from research/cad_reference, server on :8790)
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(process.argv[2] || "/node-tools/node_modules/");
const { chromium } = require("playwright");

const parts = JSON.parse(fs.readFileSync("data/parts/parts.json", "utf8")).parts;
let s = 12345;
const R = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296, U = (a, b) => a + (b - a) * R();
let jobs = [];
const add = (stl, label, n, o = {}) => {
  for (let i = 0; i < n; i++) jobs.push({ id: `${label.set}_${String(jobs.length).padStart(4, "0")}`, stl: `../data/parts/${stl}.stl`, ...label,
    seed: (R() * 2 ** 31) | 0, rot: U(0, 360), tilt: U(o.tmin ?? 0, o.tmax ?? 10), tiltAz: U(0, 360), tx: U(-8, 8), ty: U(-8, 8),
    dist: U(215, 270), stain: !!o.stain, flip: !!o.flip, part: stl });
};
add("ref", { set: "good", kind: "good" }, 60);
add("ref", { set: "dirty", kind: "good" }, 20, { stain: true });
for (const [name, meta] of Object.entries(parts)) if (meta.kind === "defect") add(name, { set: "defect", kind: "defect", defect: meta.defect }, 5);
for (const name of ["other_flange", "other_impeller5", "other_plate"]) add(name, { set: "other", kind: "other", other: parts[name].other }, 20);
add("ref", { set: "other", kind: "other", other: "bottom_face" }, 20, { flip: true });
add("ref", { set: "handheld", kind: "good" }, 20, { tmin: 10, tmax: 20 });
add("other_impeller5", { set: "handheld", kind: "other", other: "impeller_5_blade" }, 10, { tmin: 10, tmax: 20 });
add("def_throughhole_0", { set: "handheld", kind: "defect", defect: "through_blowhole" }, 10, { tmin: 10, tmax: 20 });

const OUT = process.env.OUT || "data/photos";
if (process.env.JOBS) jobs = JSON.parse(fs.readFileSync(process.env.JOBS, "utf8"));
fs.mkdirSync(OUT, { recursive: true });
const b = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await b.newPage({ viewport: { width: 512, height: 512 } });
page.on("pageerror", e => console.log("[pageerror]", e.message));
await page.goto("http://127.0.0.1:8790/photo/photo.html");
await page.waitForFunction(() => window.ready === true);
const t0 = Date.now();
const todo = process.env.LIMIT ? jobs.filter((_, i) => i % Math.ceil(jobs.length / +process.env.LIMIT) === 0) : jobs;
for (const [i, j] of todo.entries()) {
  const url = await page.evaluate(p => window.render(p), j);
  fs.writeFileSync(`${OUT}/${j.id}.png`, Buffer.from(url.split(",")[1], "base64"));
  if (i % 40 === 0) console.log(`${i}/${todo.length} ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
if (!process.env.LIMIT) fs.writeFileSync(`${OUT}/labels.json`, JSON.stringify(jobs, null, 1));
console.log(`rendered ${todo.length} photos in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
await b.close();

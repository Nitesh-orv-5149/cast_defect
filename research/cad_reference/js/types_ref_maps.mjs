// Reference-images tab: enrol good photos in the real app, then dump the anomaly map of every test photo.
// usage: DIR=photos_station OUTF=station_maps.json CDN=<dir with onnxruntime-web, html2canvas, jspdf npm packs> node types_ref_maps.mjs
// (app served on :8765 from web/; CDN is only needed where jsdelivr is unreachable)
import { createRequire } from "module";
const require = createRequire(process.env.PW || "/node-tools/node_modules/");
const { chromium } = require("playwright");
import fs from "fs";
const S = new URL(".", import.meta.url).pathname, CDN = process.env.CDN, D = S + "../data/" + (process.env.DIR || "photos_station") + "/", FAM = process.env.FAM;
const rows = JSON.parse(fs.readFileSync(D + "labels.json")).filter(r => !FAM || r.family === FAM), good = rows.filter(r => r.kind === "good");
const enrol = good.slice(0, 10), test = [...good.slice(10), ...rows.filter(r => r.kind !== "good")];
const b = await chromium.launch({ args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] });
const ctx = await b.newContext({ viewport: { width: 1280, height: 1000 }, serviceWorkers: "block" });
await ctx.grantPermissions(["camera"], { origin: "http://127.0.0.1:8765" });
const map = { "onnxruntime-web@1.20.1/dist/": CDN + "/onnxruntime-web-1.20.1/package/dist/", "html2canvas@1.4.1/dist/": CDN + "/html2canvas-1.4.1/package/dist/", "jspdf@2.5.1/dist/": CDN + "/jspdf-2.5.1/package/dist/" };
if (CDN) await ctx.route("https://cdn.jsdelivr.net/npm/**", r => { const u = r.request().url().split("/npm/")[1];
  for (const [k, v] of Object.entries(map)) if (u.startsWith(k)) return r.fulfill({ path: v + u.slice(k.length), contentType: u.endsWith(".wasm") ? "application/wasm" : "text/javascript", headers: { "access-control-allow-origin": "*" } });
  r.abort(); });
const p = await ctx.newPage();
p.on("pageerror", e => console.log("[pageerror]", e.message));
await p.goto("http://127.0.0.1:8765/index.html");
await p.waitForFunction(() => !document.getElementById("inspect").disabled, null, { timeout: 60000 });
await p.click('#nav [data-tab="ref"]'); await p.waitForTimeout(500);
await p.fill("#pname", "Station impeller"); if (process.env.ANYROT !== "0") await p.check("#anyrot");
await p.setInputFiles("#upgood", enrol.map(r => D + r.id + ".jpg"));
await p.click("#finish");
await p.waitForFunction(() => /Ready|Inspect|reference/i.test(document.getElementById("verdict").textContent) && !document.getElementById("inspect").disabled, null, { timeout: 300000 });
console.log("enrolled:", await p.innerText("#verdict"));
const out = [];
for (const r of test) {
  const url = "data:image/jpeg;base64," + fs.readFileSync(D + r.id + ".jpg").toString("base64");
  const res = await p.evaluate(async url => {
    const im = new Image(); im.src = url; await im.decode();
    const raw = crop(im, im.naturalWidth, im.naturalHeight, view.width), x = await inspectRef(raw, active);
    return { verdict: x.verdict, score: x.score, thr: x.thr, lo: x.lo, d: Array.from(x.map) };
  }, url);
  out.push({ id: r.id, what: r.what, kind: r.kind, ...res });
  console.log(r.id, res.verdict, res.score.toFixed(2), res.thr.toFixed(2));
}
fs.writeFileSync(S + "/" + (process.env.OUTF || "station_maps.json"), JSON.stringify(out));
await b.close();

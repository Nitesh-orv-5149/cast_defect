// Validation harness for web/cad.js: per family build the profile, inspect every photo, print per-group verdict table.
const fs = require("fs"), jpeg = require("jpeg-js"), CAD = require("../../../web/cad.js");
const B = __dirname + "/../";
(async () => {
  const rows = JSON.parse(fs.readFileSync(B + "data/val/labels.json")), out = [], only = process.argv[2];
  const want = { good: "PASS", defect: "REJECT", other: "WRONG PART" };
  for (const fam of [...new Set(rows.map(r => r.family))].filter(f => !only || f === only)) {
    const rs = rows.filter(r => r.family === fam), buf = fs.readFileSync(B + rs[0].stl);
    const prof = await CAD.buildProfile(CAD.prepare(CAD.parseSTL(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)), "top").tris, { camDist: rs[0].cam });
    for (const r of rs) {
      const img = jpeg.decode(fs.readFileSync(B + "data/val/" + r.file), { useTArray: true, formatAsRGBA: true });
      const t = Date.now(), st = CAD.measure(prof, img.data, img.width, img.height), d = CAD.decide(st);
      out.push({ ...r, verdict: d.verdict, why: d.findings.map(f => f.type + (f.text.includes("(") ? "" : ":" + f.text)).join(","), iou: st.iou, fr: st.featureRatio, ms: Date.now() - t, stats: st.noPart ? null : { iou: st.iou, fr: st.featureRatio, m10: st.mismatch10, m15: st.mismatch15, m20: st.mismatch20,
        holes: st.holes.map(h => ({ open: +h.open.toFixed(3), a: +h.areaMm2.toFixed(1), m: +h.mouthMm2.toFixed(1) })), extra: st.extra.items.map(c => +c.areaMm2.toFixed(2)).filter(a => a >= 0.5),
        missing: st.missing.items.map(c => +c.areaMm2.toFixed(1)).filter(a => a >= 1), excess: st.excess.items.map(c => +c.areaMm2.toFixed(1)).filter(a => a >= 1) } });
    }
  }
  fs.writeFileSync(__dirname + "/" + (process.env.OUTNAME || "val_out.json"), JSON.stringify(out));
  const groups = {};
  for (const o of out) (groups[`${o.family}/${o.set === "handheld" ? "handheld/" : ""}${o.kind}/${o.what}`] ||= []).push(o);
  let tot = { good: [0, 0], bad: [0, 0] };
  for (const [g, os] of Object.entries(groups).sort()) {
    const c = v => os.filter(o => o.verdict === v).length, right = os.filter(o => o.verdict === want[o.kind]).length;
    const reasons = {}; for (const o of os) if (o.verdict !== want[o.kind]) for (const w of o.why.split(",").filter(Boolean)) reasons[w] = (reasons[w] || 0) + 1;
    console.log(`${g.padEnd(46)} ${String(os.length).padStart(3)}  ${(100 * right / os.length).toFixed(0).padStart(4)}%  P${String(c("PASS")).padStart(3)} R${String(c("REJECT")).padStart(3)} W${String(c("WRONG PART")).padStart(3)}  ${Object.keys(reasons).length ? JSON.stringify(reasons) : ""}`);
    const key = os[0].kind === "good" ? "good" : "bad"; tot[key][0] += os.filter(o => key === "good" ? o.verdict === "PASS" : o.verdict !== "PASS").length; tot[key][1] += os.length;
  }
  console.log(`\ngood parts passed: ${tot.good[0]}/${tot.good[1]} (${(100 * tot.good[0] / tot.good[1]).toFixed(1)}%) | bad parts kept out: ${tot.bad[0]}/${tot.bad[1]} (${(100 * tot.bad[0] / tot.bad[1]).toFixed(1)}%) | median ${out.map(o => o.ms).sort((a, b) => a - b)[out.length >> 1]} ms/photo`);
})();

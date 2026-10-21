// Bearing test kit: run web/cad.js on the rendered bearing photos (data/photos_bearing) and grade the verdicts
const fs = require("fs"), jpeg = require("jpeg-js"), CAD = require("../../../web/cad.js");
const D = __dirname + "/../data/photos_bearing/";
(async () => {
  const buf = fs.readFileSync(__dirname + "/../testkit_bearing/bearing_6004_2rs.stl");
  const prof = await CAD.buildProfile(CAD.prepare(CAD.parseSTL(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)), "top").tris, { camDist: 189 });
  const want = { good: "PASS", defect: "REJECT", other: "WRONG PART" }; let ok = 0;
  const rows = JSON.parse(fs.readFileSync(D + "labels.json"));
  for (const r of rows) {
    const img = jpeg.decode(fs.readFileSync(D + r.id + ".jpg"), { useTArray: true, formatAsRGBA: true });
    const st = CAD.measure(prof, img.data, img.width, img.height), d = CAD.decide(st), hit = d.verdict === want[r.kind]; ok += hit;
    console.log(`${r.what.padEnd(16)} ${d.verdict.padEnd(10)} ${hit ? "OK  " : "MISS"} iou ${st.iou?.toFixed(3)} fr ${st.featureRatio?.toFixed(2)} | ${d.findings.map(f => f.text).join(" | ")}`);
  }
  console.log(`${ok}/${rows.length}`);
})();

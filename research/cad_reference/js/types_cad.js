// CAD tab: is the type of each detected defect right? Labelled validation defects in ../data/val (verdict REJECT only).
// usage: node types_cad.js
const fs = require("fs"), jpeg = require("jpeg-js"), CAD = require("../../../web/cad.js");
const B = __dirname + "/../";
const EXPECT = { through_blowhole: /blowhole|pinhole/, blocked_hole: /blocked hole|hole not formed/, bore_not_cored: /hole not formed|blocked hole/,
  chipped_rim: /chipped edge/, chipped_corner: /chipped edge/, notch_in_rim: /chipped edge|broken/, broken_blade: /broken|misrun/, crack: /crack/, blind_blowhole: /blowhole/ };
(async () => {
  const rows = JSON.parse(fs.readFileSync(B + "data/val/labels.json")).filter(r => r.kind === "defect" && r.set !== "handheld"), out = [];
  for (const fam of [...new Set(rows.map(r => r.family))]) {
    const rs = rows.filter(r => r.family === fam), buf = fs.readFileSync(B + rs[0].stl);
    const prof = await CAD.buildProfile(CAD.prepare(CAD.parseSTL(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)), "top").tris, { camDist: rs[0].cam });
    for (const r of rs) {
      const img = jpeg.decode(fs.readFileSync(B + "data/val/" + r.file), { useTArray: true, formatAsRGBA: true });
      const st = CAD.measure(prof, img.data, img.width, img.height), d = CAD.decide(st);
      out.push({ g: `${fam}/${r.what}`, what: r.what, verdict: d.verdict, types: d.findings.map(f => f.text) });
    }
  }
  const groups = {};
  for (const o of out) (groups[o.g] ||= []).push(o);
  let T = [0, 0];
  for (const [g, os] of Object.entries(groups)) {
    const rej = os.filter(o => o.verdict === "REJECT"), ok = rej.filter(o => EXPECT[o.what].test(o.types[0] || ""));
    T[0] += ok.length; T[1] += rej.length;
    const seen = {}; for (const o of rej) { const k = (o.types[0] || "").split(",")[0]; seen[k] = (seen[k] || 0) + 1; }
    console.log(`${g.padEnd(30)} detected ${rej.length}/${os.length}  type right ${ok.length}/${rej.length}  ${JSON.stringify(seen)}`);
  }
  console.log(`\nprimary type correct on detected defects: ${T[0]}/${T[1]}`);
  fs.writeFileSync(__dirname + "/types_cad.json", JSON.stringify(out, null, 1));
})();

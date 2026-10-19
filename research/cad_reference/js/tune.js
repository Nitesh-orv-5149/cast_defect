// Replay decision rules on saved measurements (no image processing): node tune.js <dir> file1.json file2.json ...
const fs = require("fs");
const dir = __dirname, rows = process.argv.slice(2).flatMap(f => JSON.parse(fs.readFileSync(dir + "/" + f)));
const want = { good: "PASS", defect: "REJECT", other: "WRONG PART" };

function decide(s, c) {
  if (!s) return "NO PART";
  // a hole is checkable for "blocked" only if the CAD predicts a real see-through area from this view
  const chk = s.holes.filter(h => h.a >= c.seeMin && h.a >= c.seeFrac * (h.m || h.a));
  const blocked = chk.filter(h => h.open < c.open).length, holeArea = s.holes.reduce((a, h) => a + h.a, 0) || 1;
  const extra = s.extra.filter(a => a >= c.extra), extraArea = extra.reduce((a, b) => a + b, 0);
  const pattern = blocked >= Math.max(2, Math.ceil(0.5 * chk.length)) || extraArea > 0.5 * holeArea;
  const wrongOutline = c.gate === "iou" ? s.iou < c.iou : s[c.gate] > c.mis;
  if (wrongOutline || s.fr < c.fr || pattern) return "WRONG PART";
  return blocked || extra.length || s.missing.some(a => a >= c.outline) || s.excess.some(a => a >= c.outline) ? "REJECT" : "PASS";
}
function score(c, show) {
  const g = {};
  for (const r of rows) (g[`${r.family}/${r.set === "handheld" ? "handheld/" : ""}${r.kind}/${r.what}`] ||= []).push(r);
  let good = [0, 0], bad = [0, 0], label = [0, 0];
  for (const [k, rs] of Object.entries(g).sort()) {
    const v = rs.map(r => decide(r.stats, c)), ok = rs.filter((r, i) => v[i] === want[r.kind]).length;
    if (show) console.log(`${k.padEnd(46)} ${String(rs.length).padStart(3)} ${(100 * ok / rs.length).toFixed(0).padStart(4)}%  P${v.filter(x => x === "PASS").length} R${v.filter(x => x === "REJECT").length} W${v.filter(x => x === "WRONG PART").length}`);
    for (const [i, r] of rs.entries()) {
      if (r.set === "handheld") continue;  // station results drive the choice; handheld reported separately
      if (r.kind === "good") { good[1]++; good[0] += v[i] === "PASS"; } else { bad[1]++; bad[0] += v[i] !== "PASS"; }
      label[1]++; label[0] += v[i] === want[r.kind];
    }
  }
  return { good: good[0] / good[1], bad: bad[0] / bad[1], label: label[0] / label[1], n: [good[1], bad[1]] };
}
const base = { gate: "iou", iou: 0.9, mis: 0.05, fr: 1.45, open: 0.5, extra: 2.0, outline: 6.0, seeMin: 0, seeFrac: 0 };
const fmt = r => `good passed ${(100 * r.good).toFixed(1)}%  bad kept out ${(100 * r.bad).toFixed(1)}%  label right ${(100 * r.label).toFixed(1)}%`;
console.log("current rule (IoU gate):", fmt(score(base)));
const results = [];
for (const gate of ["iou", "m10", "m15", "m20"]) for (const mis of (gate === "iou" ? [0] : [0.02, 0.03, 0.04, 0.05, 0.06, 0.08])) for (const extra of [2, 3, 4])
for (const seeFrac of [0, 0.2, 0.3, 0.45]) for (const seeMin of [0, 3, 6]) {
  const c = { ...base, gate, mis, extra, seeFrac, seeMin }, r = score(c);
  results.push({ c, r, j: r.good + r.bad + 0.5 * r.label });
}
results.sort((a, b) => b.j - a.j);
for (const { c, r } of results.slice(0, 10)) console.log(`${c.gate} mismatch>${c.mis} extra>=${c.extra}mm² see>=${c.seeFrac}x mouth & >=${c.seeMin}mm²: ${fmt(r)}`);
if (process.env.SHOW) { const best = results[0].c; console.log("\nbest rule per group:", JSON.stringify(best)); score(best, true); console.log("\ncurrent rule per group:"); score(base, true); }

// Render jobs for the defect-type checks (run from research/cad_reference, then render with
// JOBS=data/<name>.json OUT=data/<dir> node photo/render_photos.mjs and finish with eval_m0.camera_finish to .jpg).
//   types_jobs.json        CAD tab: through crack, flash, misrun, shrinkage (make_types.py), random lighting
//   station_jobs.json      reference-images tab: one fixed station (lightSeed), 16 good + every defect family
//   station_twin_jobs.json the station defects re-rendered as good parts (same pose and light) for ground-truth location
//   blind_jobs.json        a second station and a bracket, never used for tuning
import fs from "fs";

// one random stream per call to stream(); each make() continues it, as when the sets were first rendered
function stream(seed) {
  let s = seed;
  const R = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296, U = (a, b) => a + (b - a) * R();
  return build => {
    const out = [];
    build((pre, dir, stl, what, kind, n, o = {}) => {
      for (let i = 0; i < n; i++) out.push({ id: `${pre}_${what}_${String(out.length).padStart(3, "0")}`, stl: `../data/${dir}/${stl}.stl`, what, kind,
        ...(o.family ? { family: o.family } : {}), seed: (R() * 2 ** 31) | 0, rot: o.rot ? U(o.rot[0], o.rot[1]) : U(0, 360), tilt: U(0, o.tmax ?? 6), tiltAz: U(0, 360),
        tx: U(-6, 6), ty: U(-6, 6), dist: U(236, 248), stain: false, flip: false, ...(o.light != null ? { lightSeed: o.light } : {}) });
    });
    return out;
  };
}
const IMP = ["crack", "blindhole", "throughhole", "chip", "brokenblade", "blocked"], NEW = ["throughcrack", "flash", "misrun", "shrinkage"];
const save = (name, j) => { fs.writeFileSync(`data/${name}.json`, JSON.stringify(j)); console.log(name, j.length); };

const make = stream(777);
save("types_jobs", make(add => { for (const d of NEW) for (let v = 0; v < 2; v++) add("types", "parts_types", `def_${d}_${v}`, d, "defect", 4); }));
const st = make(add => {
  add("st", "parts", "ref", "good", "good", 16, { light: 4242, tmax: 3 });
  for (const d of IMP) for (let v = 0; v < 4; v++) add("st", "parts", `def_${d}_${v}`, d, "defect", 2, { light: 4242, tmax: 3 });
  for (const d of NEW) for (let v = 0; v < 2; v++) add("st", "parts_types", `def_${d}_${v}`, d, "defect", 2, { light: 4242, tmax: 3 });
});
save("station_jobs", st);
save("station_twin_jobs", st.filter(j => j.kind !== "good").map(j => ({ ...j, id: j.id + "_twin", stl: "../data/parts/ref.stl" })));
save("blind_jobs", stream(99173)(add => {
  add("bi", "parts", "ref", "good", "good", 14, { light: 9001, tmax: 3, family: "bi" });
  for (const d of IMP) for (let v = 0; v < 4; v++) add("bi", "parts", `def_${d}_${v}`, d, "defect", 1, { light: 9001, tmax: 3, family: "bi" });
  for (const d of NEW) for (let v = 0; v < 2; v++) add("bi", "parts_types", `def_${d}_${v}`, d, "defect", 1, { light: 9001, tmax: 3, family: "bi" });
  add("bb", "parts_general/bracket", "ref", "good", "good", 14, { light: 5150, tmax: 3, rot: [-8, 8], family: "bb" });
  for (const d of ["blocked", "throughhole", "chip"]) for (let v = 0; v < 2; v++) add("bb", "parts_general/bracket", `def_${d}_${v}`, d, "defect", 3, { light: 5150, tmax: 3, rot: [-8, 8], family: "bb" });
}));

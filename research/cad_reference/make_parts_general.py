"""Generalisation test parts: an asymmetric L-bracket and a tall pump-housing-like part, each with defects and
look-alike wrong parts. Writes data/parts_general/<family>/*.stl and parts.json. Every STL is centred on its footprint."""
import json
import math
import random
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

from make_parts import SEG, cyl

OUT = Path(__file__).parent / "data" / "parts_general"


def box(x0, y0, x1, y1, z0, z1):
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def slot(cx, cy, length, r, z0=-10, h=40):
    cs = CrossSection.circle(r, SEG).translate([-length / 2, 0]) + CrossSection.circle(r, SEG).translate([length / 2, 0]) \
        + CrossSection.square([length, 2 * r]).translate([-length / 2, -r])
    return Manifold.extrude(cs, h).translate([cx, cy, z0])


def bracket(holes=((15, 15), (65, 15), (15, 55)), with_slot=True):
    m = box(0, 0, 80, 30, 0, 8) + box(0, 0, 30, 70, 0, 8)
    m += box(26, 26, 30, 56, 0, 20) + box(26, 26, 56, 30, 0, 20)  # two ribs in the inner corner
    for x, y in holes:
        m += cyl(14, 8, x, y, 0)
    for x, y in holes:
        m -= cyl(40, 4, x, y, -10)
    if with_slot:
        m -= slot(45, 12, 12, 3.5)
    return m


def housing(bolts=4, bore=18, body=28):
    m = box(-45, -45, 45, 45, 0, 10) + cyl(35, body, 0, 0, 0)
    for k in range(bolts):
        a = math.pi / 4 + 2 * math.pi * k / bolts
        m -= cyl(40, 5, 35 * math.sqrt(2) * math.cos(a), 35 * math.sqrt(2) * math.sin(a), -10)
    return m - cyl(80, bore, 0, 0, -10)


def save(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    lo, hi = tm.bounds
    tm.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    assert tm.is_watertight, path
    tm.export(path)


def main():
    rng = random.Random(11)
    fams = {}
    # --- bracket
    d = OUT / "bracket"; d.mkdir(parents=True, exist_ok=True)
    ref = bracket(); idx = {"ref": {"kind": "good"}}
    save(ref, d / "ref.stl")
    for v in range(2):
        save(ref + cyl(8, 4.2, *[(15, 15), (65, 15), (15, 55)][v], 0), d / f"def_blocked_{v}.stl"); idx[f"def_blocked_{v}"] = {"kind": "defect", "defect": "blocked_hole"}
        x, y = [(45, 22), (12, 38)][v]
        save(ref - cyl(40, rng.uniform(1.8, 2.4), x, y, -10), d / f"def_throughhole_{v}.stl"); idx[f"def_throughhole_{v}"] = {"kind": "defect", "defect": "through_blowhole"}
        cx, cy = [(80, 0), (0, 70)][v]
        save(ref - Manifold.sphere(rng.uniform(6, 8), 48).translate([cx, cy, 4]), d / f"def_chip_{v}.stl"); idx[f"def_chip_{v}"] = {"kind": "defect", "defect": "chipped_corner"}
    save(bracket().mirror([1, 0, 0]), d / "other_mirrored.stl"); idx["other_mirrored"] = {"kind": "other", "other": "mirrored_bracket"}
    save(bracket(with_slot=False), d / "other_noslot.stl"); idx["other_noslot"] = {"kind": "other", "other": "bracket_without_slot"}
    save(bracket(holes=((15, 15), (65, 15))), d / "other_2holes.stl"); idx["other_2holes"] = {"kind": "other", "other": "bracket_2_bosses"}
    fams["bracket"] = idx
    # --- housing
    d = OUT / "housing"; d.mkdir(parents=True, exist_ok=True)
    ref = housing(); idx = {"ref": {"kind": "good"}}
    save(ref, d / "ref.stl")
    pts = [(35, 35), (-35, -35)]
    for v in range(2):
        save(ref + cyl(10, 5.2, *pts[v], 0), d / f"def_blocked_{v}.stl"); idx[f"def_blocked_{v}"] = {"kind": "defect", "defect": "blocked_hole"}
        x, y = [(38, 0), (0, -38)][v]
        save(ref - cyl(40, rng.uniform(1.8, 2.4), x, y, -10), d / f"def_throughhole_{v}.stl"); idx[f"def_throughhole_{v}"] = {"kind": "defect", "defect": "through_blowhole"}
        cx, cy = [(45, -45), (-45, 45)][v]
        save(ref - Manifold.sphere(rng.uniform(7, 9), 48).translate([cx, cy, 5]), d / f"def_chip_{v}.stl"); idx[f"def_chip_{v}"] = {"kind": "defect", "defect": "chipped_corner"}
    save(housing(bore=22), d / "other_bigbore.stl"); idx["other_bigbore"] = {"kind": "other", "other": "bore_22mm_not_18"}
    save(housing(body=33), d / "other_fatbody.stl"); idx["other_fatbody"] = {"kind": "other", "other": "body_33mm_not_28"}
    save(housing(bolts=2), d / "other_2bolts.stl"); idx["other_2bolts"] = {"kind": "other", "other": "2_bolt_holes"}
    fams["housing"] = idx
    json.dump(fams, open(OUT / "parts.json", "w"), indent=1)
    print({k: len(v) for k, v in fams.items()})


if __name__ == "__main__":
    main()

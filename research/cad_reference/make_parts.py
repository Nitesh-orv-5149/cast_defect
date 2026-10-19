"""Generate the synthetic CAD set for M0: a reference impeller casting, defective variants and other castings.

Writes STL files (mm) to research/cad_reference/data/parts/ plus parts.json describing each file.
The reference part is also copied to research/cad_reference/impeller.stl as a sample CAD for the app.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

HERE = Path(__file__).parent
OUT = HERE / "data" / "parts"
SEG = 96  # circle segments


def cyl(h, r, x=0.0, y=0.0, z=0.0):
    return Manifold.cylinder(h, r, -1.0, SEG).translate([x, y, z])


def blade(a0, r0, r1, sweep, t, z0, h):
    """Curved blade: band of thickness t along a spiral-ish centreline from radius r0 to r1."""
    rs = np.linspace(r0, r1, 24)
    ang = a0 + sweep * ((rs - r0) / (r1 - r0)) ** 1.2
    c = np.stack([rs * np.cos(ang), rs * np.sin(ang)], 1)
    d = np.gradient(c, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    poly = np.concatenate([c + n * t / 2, (c - n * t / 2)[::-1]])
    if 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]) < 0:
        poly = poly[::-1]
    return Manifold.extrude(CrossSection([poly]), h).translate([0, 0, z0])


def blade_angle(i, n, r, r0=18.0, r1=47.0, sweep=math.radians(35)):
    return 2 * math.pi * i / n + sweep * ((r - r0) / (r1 - r0)) ** 1.2


def impeller(blades=6, holes=3, hole_r=2.5, hole_R=31.0, bore=6.0):
    m = cyl(6, 50) + cyl(16, 16)
    for i in range(blades):
        m += blade(2 * math.pi * i / blades, 18, 47, math.radians(35), 3.5, 5.9, 7.1)
    m -= cyl(40, bore, z=-10)
    hole_xy = []
    for k in range(holes):
        # balance holes sit mid-gap between two blades at radius hole_R
        i = k * blades // holes
        a = blade_angle(i, blades, hole_R) + math.pi / blades
        x, y = hole_R * math.cos(a), hole_R * math.sin(a)
        m -= cyl(40, hole_r, x, y, -10)
        hole_xy.append((x, y))
    return m, hole_xy


def gap_point(rng, blades=6, rmin=24, rmax=44):
    """A point on the disc top between blades, away from the designed holes."""
    r = rng.uniform(rmin, rmax)
    i = rng.randrange(blades)
    a = blade_angle(i, blades, r) + math.pi / blades + rng.uniform(-0.12, 0.12)
    return r * math.cos(a), r * math.sin(a)


def save(m, name, meta, index):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    assert tm.is_watertight, name
    tm.export(OUT / f"{name}.stl")
    index[name] = meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    index = {}
    ref, holes = impeller()
    save(ref, "ref", {"kind": "good"}, index)
    trimesh.load(OUT / "ref.stl").export(HERE / "impeller.stl")

    for v in range(4):
        x, y = gap_point(rng)
        save(ref - cyl(40, rng.uniform(1.6, 2.4), x, y, -10), f"def_throughhole_{v}", {"kind": "defect", "defect": "through_blowhole", "xy": [x, y]}, index)
        pits = [gap_point(rng) for _ in range(rng.randint(1, 3))]
        m = ref
        for x, y in pits:
            m -= Manifold.sphere(rng.uniform(1.8, 3.0), 48).translate([x, y, 6.2])
        save(m, f"def_blindhole_{v}", {"kind": "defect", "defect": "blind_blowhole", "xy": pits}, index)
        hx, hy = holes[v % len(holes)]
        save(ref + cyl(6, 2.6, hx, hy, 0), f"def_blocked_{v}", {"kind": "defect", "defect": "blocked_hole", "xy": [hx, hy]}, index)
        a = rng.uniform(0, 2 * math.pi)
        save(ref - Manifold.sphere(rng.uniform(5, 8), 48).translate([50 * math.cos(a), 50 * math.sin(a), 3]), f"def_chip_{v}",
             {"kind": "defect", "defect": "chipped_rim", "angle": a}, index)
        # remove the outer part of one blade (above the disc) with a wedge that follows the blade's sweep
        i, rc = rng.randrange(6), rng.uniform(28, 38)
        a0, a1 = blade_angle(i, 6, rc) - 0.12, blade_angle(i, 6, 47) + 0.12
        fan = [(0.0, 0.0)] + [(60 * math.cos(a), 60 * math.sin(a)) for a in np.linspace(a0, a1, 16)]
        wedge = Manifold.extrude(CrossSection([np.array(fan)]), 20).translate([0, 0, 6.05])
        cut = wedge - cyl(40, rc, z=-10)
        save(ref - cut, f"def_brokenblade_{v}", {"kind": "defect", "defect": "broken_blade", "blade": i, "r_cut": rc}, index)
        x, y = gap_point(rng, rmin=26, rmax=40)
        crack = Manifold.cube([rng.uniform(10, 16), 0.7, 4], True).rotate([0, 0, rng.uniform(0, 180)]).translate([x, y, 6])
        save(ref - crack, f"def_crack_{v}", {"kind": "defect", "defect": "crack", "xy": [x, y]}, index)

    flange = cyl(8, 50) - cyl(40, 20, z=-10)
    for k in range(8):
        a = 2 * math.pi * k / 8
        flange -= cyl(40, 3.5, 38 * math.cos(a), 38 * math.sin(a), -10)
    save(flange, "other_flange", {"kind": "other", "other": "flange"}, index)
    imp5, _ = impeller(blades=5, holes=2, hole_R=33, bore=8)
    save(imp5, "other_impeller5", {"kind": "other", "other": "impeller_5_blade"}, index)
    plate = Manifold.cube([90, 90, 6]).translate([-45, -45, 0]) - cyl(40, 10, z=-10)
    for sx in (-1, 1):
        for sy in (-1, 1):
            plate -= cyl(40, 4, 33 * sx, 33 * sy, -10)
    save(plate, "other_plate", {"kind": "other", "other": "square_plate"}, index)
    json.dump({"holes": holes, "parts": index}, open(OUT / "parts.json", "w"), indent=1)
    print(f"wrote {len(index)} parts to {OUT}")


if __name__ == "__main__":
    main()

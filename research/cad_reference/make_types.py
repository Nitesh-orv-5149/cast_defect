"""Extra impeller defects for checking the defect-type labels: the shapes the M0 set lacks in the CAD view
(through crack, flash fin, misrun, shrinkage cavity). Writes STL files to data/parts_types/ plus parts.json.
"""
import json
import math
import random

import numpy as np
from manifold3d import CrossSection, Manifold

from make_parts import HERE, cyl, gap_point, impeller, save
import make_parts

OUT = HERE / "data" / "parts_types"


def sector(r0, r1, a0, a1, z0, h):
    a = np.linspace(a0, a1, 32)
    poly = np.concatenate([np.stack([r1 * np.cos(a), r1 * np.sin(a)], 1), np.stack([r0 * np.cos(a), r0 * np.sin(a)], 1)[::-1]])
    return Manifold.extrude(CrossSection([poly]), h).translate([0, 0, z0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_parts.OUT = OUT
    rng = random.Random(11)
    index = {}
    ref, _ = impeller()
    for v in range(2):
        x, y = gap_point(rng, rmin=28, rmax=40)
        slot = Manifold.cube([rng.uniform(12, 16), 1.6, 40], True).rotate([0, 0, rng.uniform(0, 180)]).translate([x, y, 0])
        save(ref - slot, f"def_throughcrack_{v}", {"kind": "defect", "defect": "through_crack"}, index)
        a = rng.uniform(0, 2 * math.pi)
        save(ref + sector(49.5, 55.5, a, a + math.radians(rng.uniform(35, 50)), 2.5, 1.0), f"def_flash_{v}", {"kind": "defect", "defect": "flash"}, index)
        a = rng.uniform(0, 2 * math.pi)
        save(ref - cyl(40, rng.uniform(16, 19), 56 * math.cos(a), 56 * math.sin(a), -10), f"def_misrun_{v}", {"kind": "defect", "defect": "misrun"}, index)
        x, y = gap_point(rng, rmin=30, rmax=40)
        # shrinkage: a ragged, branching void rather than a round gas hole
        cav = cyl(40, 1.4, x, y, -10)
        for dx, dy, r in [(2.6, 0.6, 1.1), (4.8, 1.8, 0.9), (0.8, 2.8, 1.0), (1.5, 5.0, 0.8), (-2.2, 1.4, 0.9)]:
            cav += cyl(40, r, x + dx, y + dy, -10)
        save(ref - cav, f"def_shrinkage_{v}", {"kind": "defect", "defect": "shrinkage_cavity"}, index)
    json.dump({"parts": index}, open(OUT / "parts.json", "w"), indent=1)
    print("wrote", sorted(index))


if __name__ == "__main__":
    main()

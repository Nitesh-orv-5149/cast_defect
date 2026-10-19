"""Blind test kit: a connecting-rod casting nothing was tuned on, with defects and wrong parts.

Writes the CAD to research/cad_reference/testkit/ (STL, OBJ, GLB) and the variants to data/testkit_parts/.
Shapes (mm): big-end ring Ø56 with Ø36 bore, small-end ring Ø28 with Ø16 bore, centres 110 apart, two bolt lugs
with Ø7 through-holes at the big end, I-beam shank with a 4 mm deep blind pocket on the top face.
"""
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

from make_parts import SEG, cyl

HERE = Path(__file__).parent
KIT = HERE / "testkit"
VAR = HERE / "data" / "testkit_parts"


def taper(x0, w0, x1, w1, z0, h):
    poly = np.array([[x0, -w0 / 2], [x1, -w1 / 2], [x1, w1 / 2], [x0, w0 / 2]])
    return Manifold.extrude(CrossSection([poly]), h).translate([0, 0, z0])


def conrod(L=110.0, pocket=True, lugs=True, small_bore=True):
    m = cyl(20, 28) + cyl(18, 14, L, 0, 1) + taper(18, 26, L - 10, 16, 3, 14)
    if lugs:
        for s in (-1, 1):  # bolt lug: boss plus a web joining it to the big-end ring
            m += cyl(20, 7.5, 6, s * 31) + Manifold.cube([12, 10, 20]).translate([0, s * 26 - 5, 0])
    if pocket:
        m -= taper(26, 18, L - 18, 9, 13, 10)  # blind I-beam pocket, floor at z = 13 (4 mm deep)
    m -= cyl(60, 18, 0, 0, -10)
    if small_bore:
        m -= cyl(60, 8, L, 0, -10)
    if lugs:
        for s in (-1, 1):
            m -= cyl(60, 3.5, 6, s * 31, -10)
    return m


def save(m, path):
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    lo, hi = tm.bounds
    tm.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    assert tm.is_watertight, path
    tm.export(path)
    return tm


def main():
    KIT.mkdir(exist_ok=True)
    VAR.mkdir(parents=True, exist_ok=True)
    ref = conrod()
    tm = save(ref, KIT / "conrod.stl")
    tm.export(KIT / "conrod.obj")
    tm.export(KIT / "conrod.glb")
    save(ref, VAR / "ref.stl")
    # defects (offsets are in the CAD frame before recentring; the part spans x = -28..124)
    save(ref + cyl(20, 3.7, 6, 31), VAR / "def_blocked_lug.stl")
    save(ref - cyl(60, 2.2, 62, 2, -10), VAR / "def_blowhole_in_pocket.stl")
    save(ref - Manifold.sphere(9, 48).translate([-27, -8, 10]), VAR / "def_chipped_big_end.stl")
    save(conrod(small_bore=False), VAR / "def_small_bore_not_cored.stl")
    # wrong parts
    save(conrod(L=95), VAR / "other_short_rod.stl")
    save(conrod(lugs=False), VAR / "other_no_lugs.stl")
    print("testkit CAD:", sorted(p.name for p in KIT.iterdir()), "| variants:", sorted(p.stem for p in VAR.glob("*.stl")))


if __name__ == "__main__":
    main()

"""CAD of a 6004-2RS deep-groove ball bearing (20 x 42 x 12 mm, rubber seals both sides) for the Reference CAD tab.

Dimensions: bore d = 20, outside D = 42, width B = 12, outer-ring shoulder D2 = 37.2, inner-ring shoulder d2 = 24.9,
chamfer r = 0.6 (catalogue values for 6004-2RS). The seals sit 0.4 mm below the ring faces, so the camera sees
three steps on each face: inner ring, seal, outer ring. Writes testkit_bearing/bearing_6004_2rs.{stl,obj}.
"""
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

OUT = Path(__file__).parent / "testkit_bearing"
d, D, B, D2, d2, r, seal = 20.0, 42.0, 12.0, 37.2, 24.9, 0.6, 0.4


def profile():
    """Half cross-section in (radius, z), counter-clockwise."""
    ri, ro, si, so = d / 2, D / 2, d2 / 2, D2 / 2
    return np.array([
        (ri + r, 0), (si, 0), (si, seal), (so, seal), (so, 0), (ro - r, 0), (ro, r),
        (ro, B - r), (ro - r, B), (so, B), (so, B - seal), (si, B - seal), (si, B), (ri + r, B), (ri, B - r), (ri, r)])


def main():
    OUT.mkdir(exist_ok=True)
    m = Manifold.revolve(CrossSection([profile()]), 256)
    mesh = m.to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    assert tm.is_watertight
    for ext in ("stl", "obj"):
        tm.export(OUT / f"bearing_6004_2rs.{ext}")
    print("bounds", tm.bounds.round(2).tolist(), "| volume", round(tm.volume, 1), "mm3 | triangles", len(tm.faces))


if __name__ == "__main__":
    main()

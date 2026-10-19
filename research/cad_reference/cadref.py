"""CAD-reference inspection, M0 prototype (CPU only).

render():   z-buffer rasteriser for CAD templates (silhouette, through-holes, depth, normals, feature edges)
segment():  part mask from a photo on a coloured mat (hue/saturation distance from the border colour)
align():    coarse template + rotation search on hole-aware silhouettes, then ECC homography refinement
inspect():  wrong-part gate, hole-by-hole check, outline check, feature-edge check
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import trimesh


# ---------------------------------------------------------------- CAD rendering

def load_mesh(path):
    m = trimesh.load(path, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def view_rotation(tilt_deg=0.0, azim_deg=0.0, flip=False):
    """Rotation taking CAD coords to view coords (camera looks along -z of the view). flip = look at the bottom face."""
    R = np.eye(3)
    if flip:
        R = np.diag([1.0, -1.0, -1.0])
    a, t = math.radians(azim_deg), math.radians(tilt_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return Rz.T @ Rx @ Rz @ R  # tilt about an axis at angle `azim`


def render(V, F, R, px_per_mm, size, crease_deg=25.0):
    """Orthographic z-buffer render. Returns mask (part pixels), holes (background enclosed by the part),
    depth (mm, view z, larger = closer), normal (view space), edges (outline + creases + steps)."""
    P = V @ R.T
    c = (size - 1) / 2
    x, y, z = c + P[:, 0] * px_per_mm, c - P[:, 1] * px_per_mm, P[:, 2]
    tri = F
    x0, x1, x2 = x[tri[:, 0]], x[tri[:, 1]], x[tri[:, 2]]
    y0, y1, y2 = y[tri[:, 0]], y[tri[:, 1]], y[tri[:, 2]]
    n = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    depth = np.full((size, size), -np.inf)
    fid = np.full((size, size), -1, np.int64)
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    for i in np.nonzero((n[:, 2] > 1e-6) & (np.abs(den) > 1e-9))[0]:
        xa, xb = int(max(0, math.floor(min(x0[i], x1[i], x2[i])))), int(min(size - 1, math.ceil(max(x0[i], x1[i], x2[i]))))
        ya, yb = int(max(0, math.floor(min(y0[i], y1[i], y2[i])))), int(min(size - 1, math.ceil(max(y0[i], y1[i], y2[i]))))
        if xa > xb or ya > yb:
            continue
        gx, gy = np.meshgrid(np.arange(xa, xb + 1) + 0.0, np.arange(ya, yb + 1) + 0.0)
        w0 = ((y1[i] - y2[i]) * (gx - x2[i]) + (x2[i] - x1[i]) * (gy - y2[i])) / den[i]
        w1 = ((y2[i] - y0[i]) * (gx - x2[i]) + (x0[i] - x2[i]) * (gy - y2[i])) / den[i]
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        zz = w0 * z[tri[i, 0]] + w1 * z[tri[i, 1]] + w2 * z[tri[i, 2]]
        sub = depth[ya:yb + 1, xa:xb + 1]
        upd = inside & (zz > sub)
        sub[upd] = zz[upd]
        fid[ya:yb + 1, xa:xb + 1][upd] = i
    mask = fid >= 0
    normal = np.zeros((size, size, 3))
    normal[mask] = n[fid[mask]]
    holes = enclosed_background(mask)
    # feature edges: outline, normal creases, depth steps (all inside or on the part)
    m8 = mask.astype(np.uint8)
    outline = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))) > 0
    cosv = np.cos(np.radians(crease_deg))
    dfin = np.where(mask, depth, 0.0)
    crease = np.zeros_like(mask)
    step = np.zeros_like(mask)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = normal[: size - dy, : size - dx], normal[dy:, dx:]
        ma, mb = mask[: size - dy, : size - dx], mask[dy:, dx:]
        both = ma & mb
        c_ = both & ((a * b).sum(-1) < cosv)
        s_ = both & (np.abs(dfin[: size - dy, : size - dx] - dfin[dy:, dx:]) > 0.8)
        crease[: size - dy, : size - dx] |= c_
        step[: size - dy, : size - dx] |= s_
    edges = outline | crease | step
    return {"mask": mask, "holes": holes, "depth": np.where(mask, depth, np.nan), "normal": normal, "edges": edges,
            "outline": outline}


def enclosed_background(mask):
    """Background components not touching the image border = through-holes."""
    bg = (~mask).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return np.isin(lab, [k for k in range(1, n) if k not in border])


# ---------------------------------------------------------------- photo side

def segment(img_bgr):
    """Part mask on a coloured mat. Score = how much of the mat's own hue a pixel carries, relative to its lightness:
    lit and shadowed mat score the same, grey metal (highlights included) scores ~0. Otsu splits the two."""
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = lambda a: np.concatenate([a[:6].reshape(-1, a.shape[-1]), a[-6:].reshape(-1, a.shape[-1]),
                                       a[:, :6].reshape(-1, a.shape[-1]), a[:, -6:].reshape(-1, a.shape[-1])])
    h, w = lab.shape[:2]
    ab, light = lab[..., 1:] - 128, lab[..., 0] + 20
    u = np.median(border(ab), 0)
    u /= np.linalg.norm(u) + 1e-6
    s = (ab @ u) / light
    s_mat = float(np.median(border(s[..., None])))
    d8 = cv2.normalize(np.clip(s_mat - s, 0, None), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    part = lab_ == k
    # small background specks inside the part are noise, real holes are kept (min area)
    holes = enclosed_background(part)
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
    for j in range(1, nh):
        if hst[j, cv2.CC_STAT_AREA] < 12:
            part[hl == j] = True
    return part




# ---------------------------------------------------------------- templates and alignment

@dataclass
class Profile:
    V: np.ndarray
    F: np.ndarray
    radius_mm: float
    views: list = field(default_factory=list)  # (tilt, azim, render dict)
    size: int = 320
    px_per_mm: float = 1.0
    coarse: int = 96


def build_profile(V, F, size=320, tilts=(0, 6, 12, 18), azims=8, flip=False):
    r = float(np.max(np.linalg.norm(V[:, :2], axis=1)))  # CAD origin = part axis
    ppm = (size * 0.42) / r
    prof = Profile(V, F, r, size=size, px_per_mm=ppm)
    for t in tilts:
        for a in ([0.0] if t == 0 else np.arange(azims) * 360.0 / azims):
            rd = render(V, F, view_rotation(t, a, flip), ppm, size)
            rd["raised"] = raised_regions(rd, ppm)
            prof.views.append((t, a, rd))
    return prof


def raised_regions(rd, ppm, min_height_mm=1.0):
    """Features standing above the main face (blades, hub): the CAD says an edge must bound them."""
    d = rd["depth"][rd["mask"]]
    base = np.percentile(d, 25)
    return rd["mask"] & (np.nan_to_num(rd["depth"], nan=-1e9) > base + min_height_mm)


def _centroid_radius(mask):
    ys, xs = np.nonzero(mask)
    return ys.mean(), xs.mean(), math.sqrt(mask.sum() / math.pi)


def _canon(img, cy, cx, scale, size, angle_deg, interp=cv2.INTER_LINEAR):
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += (size - 1) / 2 - cx
    M[1, 2] += (size - 1) / 2 - cy
    return cv2.warpAffine(img, M, (size, size), flags=interp, borderValue=0), M


def align(prof, photo_mask, step_deg=2.0):
    """Coarse: every template view x in-plane rotation on hole-aware silhouettes at low resolution.
    Fine: ECC homography on blurred masks at full template resolution. Returns photo->template homography."""
    S, C = prof.size, prof.coarse
    f = C / S
    cy, cx, r = _centroid_radius(photo_mask)
    pm8 = photo_mask.astype(np.uint8) * 255
    best = None
    for vi, (_, _, rd) in enumerate(prof.views):
        tcy, tcx, tr = _centroid_radius(rd["mask"])
        T = np.float32([[f, 0, (C - 1) / 2 - f * tcx], [0, f, (C - 1) / 2 - f * tcy]])
        tmc = cv2.warpAffine(rd["mask"].astype(np.uint8) * 255, T, (C, C)) > 127
        scale = f * tr / r
        for ang in np.arange(0, 360, step_deg):
            pmb = _canon(pm8, cy, cx, scale, C, ang)[0] > 127
            iou = (pmb & tmc).sum() / max((pmb | tmc).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, vi, ang, tr / r, (tcy, tcx))
    iou0, vi, ang, scale, (tcy, tcx) = best
    M = cv2.getRotationMatrix2D((cx, cy), ang, scale)
    M[0, 2] += tcx - cx
    M[1, 2] += tcy - cy
    A = np.vstack([M, [0, 0, 1]])  # photo -> template
    tgt = cv2.GaussianBlur(prof.views[vi][2]["mask"].astype(np.float32), (0, 0), 2.0)
    src = cv2.GaussianBlur(photo_mask.astype(np.float32), (0, 0), 2.0 / scale)
    H = A
    try:
        _, W = cv2.findTransformECC(tgt, src, np.linalg.inv(A).astype(np.float32), cv2.MOTION_HOMOGRAPHY,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6), None, 5)
        H = np.linalg.inv(W)  # ECC gives template->photo
    except cv2.error:
        pass
    return {"view": vi, "angle": float(ang), "scale": float(scale), "H": H, "coarse_iou": float(iou0)}


def to_template(img, H, S, interp=cv2.INTER_LINEAR):
    return cv2.warpPerspective(img, H, (S, S), flags=interp, borderValue=0)


def from_template(img, H, shape, interp=cv2.INTER_LINEAR):
    return cv2.warpPerspective(img, np.linalg.inv(H), (shape[1], shape[0]), flags=interp, borderValue=0)



# ---------------------------------------------------------------- measurements (thresholds live in decide())

def edge_normals(rd):
    """Unit normal of the CAD's edges, from the blurred depth map (background treated as a deep floor)."""
    d = rd["depth"]
    base = np.nanmin(d) - 6.0
    dd = cv2.GaussianBlur(np.where(rd["mask"], np.nan_to_num(d, nan=base), base).astype(np.float32), (0, 0), 1.5)
    nx, ny = cv2.Sobel(dd, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(dd, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(nx, ny) + 1e-6
    return nx / mag, ny / mag


def photo_gradients(gray_t, part):
    g = cv2.GaussianBlur(gray_t.astype(np.float32), (0, 0), 1.2)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    G = np.hypot(gx, gy)
    inner = cv2.erode(part.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    s = np.percentile(G[inner], 95) + 1e-6 if inner.any() else 1.0
    return gx / s, gy / s, G / s


def _components(reg, mm2, conn=8):
    n, lab, st, _ = cv2.connectedComponentsWithStats(reg.astype(np.uint8), connectivity=conn)
    return [(float(st[j, cv2.CC_STAT_AREA] * mm2), int(max(st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]))) for j in range(1, n)]


def measure(prof, img_bgr, outline_tol_mm=1.0, chunk_mm=6.0, stray_far_mm=1.5):
    """Everything the decision needs, computed once per photo (so thresholds can be tuned offline)."""
    mask = segment(img_bgr)
    if mask.sum() < 500:
        return {"no_part": True}
    al = align(prof, mask)
    S, H, ppm = prof.size, al["H"], prof.px_per_mm
    rd = prof.views[al["view"]][2]
    mm2 = 1.0 / ppm ** 2
    pm = to_template(mask.astype(np.uint8) * 255, H, S) > 127
    gray = to_template(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), H, S)
    tm, th, te, raised = rd["mask"], rd["holes"], rd["edges"], rd["raised"]
    gx, gy, G = photo_gradients(gray, pm & tm)
    nx, ny = edge_normals(rd)
    proj = np.abs(gx * nx + gy * ny)  # photo edge strength across the CAD edge direction
    P = cv2.dilate(proj, np.ones((5, 5), np.uint8))
    k3 = np.ones((3, 3), np.uint8)
    near_outline = cv2.dilate(rd["outline"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    interior_edges = te & ~near_outline & ~cv2.dilate(th.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    far = cv2.distanceTransform((~te).astype(np.uint8), cv2.DIST_L2, 3) > 4
    body = cv2.erode((tm & ~th).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    Gd = cv2.dilate(G, np.ones((5, 5), np.uint8))
    on = float(np.median(P[interior_edges])) if interior_edges.any() else 0.0
    off = float(np.median(Gd[body & far])) * 2 / math.pi if (body & far).any() else 1.0
    st = {"iou": float((pm & tm).sum() / max((pm | tm).sum(), 1)), "feature_ratio": on / max(off, 1e-6),
          "view": al["view"], "tilt": prof.views[al["view"]][0], "ppm": ppm}
    tol = max(1, int(round(outline_tol_mm * ppm)))
    kt = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    # designed through-holes: how open is each one
    nh, hl = cv2.connectedComponents(th.astype(np.uint8), connectivity=4)
    st["holes"] = []
    for j in range(1, nh):
        reg = hl == j
        core = cv2.erode(reg.astype(np.uint8), k3).astype(bool)
        core = core if core.any() else reg
        st["holes"].append({"area_mm2": float(reg.sum() * mm2), "open": float((~pm[core]).mean())})
    photo_holes = enclosed_background(pm)
    expected_bg = cv2.dilate((th | ~tm).astype(np.uint8), kt).astype(bool)
    st["extra_holes"] = [a for a, _ in _components(photo_holes & ~expected_bg, mm2, 4)]
    band = cv2.dilate(rd["outline"].astype(np.uint8), kt).astype(bool)
    st["missing"] = [a for a, _ in _components(cv2.morphologyEx((tm & ~th & ~pm & ~band & ~photo_holes).astype(np.uint8), cv2.MORPH_OPEN, k3), mm2)]
    st["excess"] = [a for a, _ in _components(cv2.morphologyEx((pm & ~tm & ~band).astype(np.uint8), cv2.MORPH_OPEN, k3), mm2)]
    # raised features in chunks: support = 90th pct of across-edge strength on the chunk's CAD edges (best side wins)
    cell = max(4, int(chunk_mm * ppm))
    nb, blab = cv2.connectedComponents(raised.astype(np.uint8), connectivity=8)
    chunks = []
    for j in range(1, nb):
        comp = blab == j
        ys, xs = np.nonzero(comp)
        for gyc in range(ys.min() // cell, ys.max() // cell + 1):
            for gxc in range(xs.min() // cell, xs.max() // cell + 1):
                ch = np.zeros_like(comp)
                ch[gyc * cell:(gyc + 1) * cell, gxc * cell:(gxc + 1) * cell] = True
                ch &= comp
                if ch.sum() < 0.12 * cell * cell:
                    continue
                ring = cv2.dilate(ch.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & te & ~near_outline
                if ring.sum() < cell:
                    continue
                chunks.append({"blade": j, "support": float(np.percentile(P[ring], 90)), "area_mm2": float(ch.sum() * mm2)})
    st["chunks"] = chunks
    # strong edges with no CAD edge nearby
    r = int(stray_far_mm * ppm)
    keep_out = cv2.dilate((te | th | ~tm).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(bool)
    stray = []
    for thr in (0.9, 1.2, 1.6):
        s = cv2.morphologyEx(((G > thr) & tm & pm & ~keep_out).astype(np.uint8), cv2.MORPH_CLOSE, k3)
        stray.append([(a, L / ppm) for a, L in _components(s, mm2)])
    st["stray"] = stray
    return st


DEFAULTS = {"iou_min": 0.90, "feature_ratio_min": 1.45, "pattern_mismatch": 0.5, "hole_open_min": 0.5,
            "extra_hole_mm2": 2.0, "outline_mm2": 6.0, "chunk_rel_min": 0.3, "stray_level": 1, "crack_len_mm": 6.0,
            "pit_area_mm2": 3.0, "use_features": True, "use_stray": True}


def decide(st, cfg=None):
    cfg = {**DEFAULTS, **(cfg or {})}
    if st.get("no_part"):
        return "NO PART", []
    holes = st["holes"]
    blocked = [h for h in holes if h["open"] < cfg["hole_open_min"]]
    hole_area = sum(h["area_mm2"] for h in holes) or 1.0
    extra = [a for a in st["extra_holes"] if a >= cfg["extra_hole_mm2"]]
    # a different hole pattern (most designed holes closed and/or lots of unexplained openings) = different part
    pattern_off = (len(blocked) >= max(2, math.ceil(cfg["pattern_mismatch"] * len(holes)))) or sum(extra) > cfg["pattern_mismatch"] * hole_area
    if st["iou"] < cfg["iou_min"] or st["feature_ratio"] < cfg["feature_ratio_min"] or pattern_off:
        return "WRONG PART", [{"type": "mismatch", "iou": st["iou"], "feature_ratio": st["feature_ratio"], "pattern": bool(pattern_off)}]
    F = [{"type": "blocked_hole", "area_mm2": h["area_mm2"]} for h in blocked]
    F += [{"type": "unexpected_hole", "area_mm2": a} for a in extra]
    F += [{"type": "missing_material", "area_mm2": a} for a in st["missing"] if a >= cfg["outline_mm2"]]
    F += [{"type": "excess_material", "area_mm2": a} for a in st["excess"] if a >= cfg["outline_mm2"]]
    if cfg["use_features"] and st["chunks"]:
        med = float(np.median([c["support"] for c in st["chunks"]]))
        weak = [c for c in st["chunks"] if c["support"] < cfg["chunk_rel_min"] * med]
        if weak:
            F.append({"type": "missing_feature", "chunks": len(weak), "area_mm2": sum(c["area_mm2"] for c in weak)})
    if cfg["use_stray"]:
        for a, L in st["stray"][cfg["stray_level"]]:
            if L >= cfg["crack_len_mm"]:
                F.append({"type": "unexpected_edge", "length_mm": L})
            elif a >= cfg["pit_area_mm2"]:
                F.append({"type": "unexpected_spot", "area_mm2": a})
    return ("REJECT" if F else "PASS"), F


def inspect(prof, img_bgr, cfg=None):
    st = measure(prof, img_bgr)
    v, F = decide(st, cfg)
    return {"verdict": v, "findings": F, "stats": st}

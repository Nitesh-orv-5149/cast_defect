"""Defective variants of the user's real 6004-2RS bearing photos, built from the photo's own pixels.

Each base is the app's framed 1024 px crop (bearing centred) with its part mask. Holes and chips show the paper
beneath, taken from just outside the bearing on the same row so the ruled lines continue; extra metal and the blocked
bore reuse the bearing's own ring texture through a polar remap.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

G = Path(sys.argv[1])
rng = np.random.default_rng(7)


def load(k):
    img = np.asarray(Image.open(G / f"base_{k}.png").convert("RGB")).astype(np.float32)
    m = np.asarray(Image.open(G / f"mask_{k}.png").convert("L").resize(img.shape[1::-1], Image.NEAREST)) > 127
    n = img.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    cy, cx = yy[m].mean(), xx[m].mean()
    filled = ndi.binary_fill_holes(m)
    bore = filled & ~m
    lab, k_ = ndi.label(bore)
    if k_:
        big = 1 + np.argmax(ndi.sum(bore, lab, range(1, k_ + 1)))
        bore = lab == big
    R = np.sqrt(filled.sum() / np.pi)
    r = np.sqrt(bore.sum() / np.pi)
    return img, m, filled, bore, cx, cy, R, r, xx, yy


def soft(mask, s=1.5):
    return np.clip(ndi.gaussian_filter(mask.astype(np.float32), s), 0, 1)[..., None]


def blend(img, src, mask, s=1.5):
    a = soft(mask, s)
    return img * (1 - a) + src * a


def paper_mirror(img, region, x_line):
    """Paper for pixels in `region`: mirrored across the vertical line x_line (outside the part, same row)."""
    out = img.copy()
    ys, xs = np.nonzero(region)
    xs2 = np.clip(np.round(2 * x_line - xs).astype(int), 0, img.shape[1] - 1)
    out[ys, xs] = img[ys, xs2]
    return out


def paper_shift(img, region, filled, side=1):
    """Paper for pixels in `region`: same row, just beyond the part's edge on the left (-1) or right (+1)."""
    out = img.copy()
    n = img.shape[1]
    for y in np.unique(np.nonzero(region)[0]):
        cols = np.nonzero(filled[y])[0]
        if not len(cols):
            continue
        edge = cols.max() + 6 if side > 0 else cols.min() - 6
        xs = np.nonzero(region[y])[0]
        off = xs - xs.min()
        src = np.clip(edge + side * (off + 2), 0, n - 1) if side > 0 else np.clip(edge - (off.max() - off) - 2, 0, n - 1)
        out[y, xs] = img[y, src]
    return out


def polar_sample(img, cx, cy, rho, th):
    xs, ys = cx + rho * np.cos(th), cy + rho * np.sin(th)
    return np.stack([ndi.map_coordinates(img[..., c], [ys, xs], order=1, mode="nearest") for c in range(3)], -1)


def jagged(mask, amp=3):
    """Roughen a mask's boundary a little so cut edges don't look machined."""
    noise = ndi.gaussian_filter(rng.standard_normal(mask.shape), 2) * amp * 6
    d_in = ndi.distance_transform_edt(mask)
    d_out = ndi.distance_transform_edt(~mask)
    sd = np.where(mask, d_in, -d_out)
    return sd + noise > 0


def rim_shade(img, region, width=4, k=0.55):
    """Darken a thin band around a new edge (fracture / cavity wall)."""
    band = ndi.binary_dilation(region, iterations=width) & ~region
    a = soft(band, 1.2) * (1 - k)
    return img * (1 - a)


def chip(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    th = np.deg2rad(185)
    bx, by = cx + R * np.cos(th), cy + R * np.sin(th)
    bite = jagged(((xx - bx) ** 2 + ((yy - by) / 0.8) ** 2) < ((0.22 if k == "02" else 0.16) * R) ** 2) & filled
    x_line = xx[filled].min() - 3
    out = blend(img, paper_mirror(img, bite, x_line), bite, 1.0)
    out = rim_shade(out, ~bite & filled & ndi.binary_dilation(bite, iterations=6), 0, 1)  # keep metal untouched
    edge = ndi.binary_dilation(bite, iterations=5) & filled & ~bite
    out = out * (1 - soft(edge, 1.0) * 0.35) + soft(edge, 1.0) * 0.35 * 200  # bright granular fracture edge
    return out, "chipped_rim"


def through_hole(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    th, rho = np.deg2rad(-25), (R + r) / 2
    hx, hy, hr = cx + rho * np.cos(th), cy + rho * np.sin(th), 0.075 * R
    hole = jagged(((xx - hx) ** 2 + (yy - hy) ** 2) < hr ** 2, 1.5) & m
    out = blend(img, paper_shift(img, hole, filled, +1), hole, 1.0)
    out = rim_shade(out, hole, 5, 0.35)
    return out, "through_blowhole"


def blocked_bore(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    race = m & (np.hypot(xx - cx, yy - cy) < r + 0.12 * R)
    base = np.median(img[race], 0) * 0.78
    tex = sum(ndi.gaussian_filter(rng.standard_normal(m.shape), s) * a for s, a in [(1, 10), (3, 9), (8, 7)])
    light = -((xx - cx) + (yy - cy)) / (r + 1) * 14 - 10 * (np.hypot(xx - cx, yy - cy) / (r + 1))
    web = np.clip(base[None, None, :] + (tex + light)[..., None], 0, 255)
    plug = ndi.binary_dilation(bore, iterations=2) & filled
    out = blend(img, web, plug, 1.5)
    ring = ndi.binary_dilation(bore, iterations=3) & ~ndi.binary_erosion(bore, iterations=3)
    return out * (1 - soft(ring, 1.5) * 0.45), "bore_blocked"


def porosity(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    out = img.copy()
    base_th = np.deg2rad(120)
    for i in range(11):
        th = base_th + rng.uniform(-0.3, 0.3)
        rho = r + rng.uniform(0.08, 0.2) * R
        px, py, pr = cx + rho * np.cos(th), cy + rho * np.sin(th), rng.uniform(9, 15)
        d = np.hypot(xx - px, yy - py)
        core = soft(d < pr, 1.0)
        lip = soft((d >= pr) & (d < pr + 2.5), 0.8)
        out = out * (1 - core * 0.9) + core * 0.9 * 18
        out = out * (1 - lip * 0.3) + lip * 0.3 * 230
    return out, "gas_porosity"


def through_crack(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    th0 = np.deg2rad(-40)
    pts, rho = [], r - 2
    th = th0
    while rho < R + 4:
        pts.append((cx + rho * np.cos(th), cy + rho * np.sin(th)))
        rho += 0.7
        th += rng.normal(0, 0.004)
    line = np.zeros(m.shape, bool)
    for x, y in pts:
        line[int(round(y)), int(round(x))] = True
    core = ndi.binary_dilation(line, iterations=2) & filled
    wall = ndi.binary_dilation(line, iterations=6) & filled & ~core
    out = blend(img, paper_shift(img, core, filled, +1), core, 0.6)
    out = out * (1 - soft(wall, 1.0) * 0.75)
    return out, "through_crack"


def lump(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    rho, th = np.hypot(xx - cx, yy - cy), np.arctan2(yy - cy, xx - cx)
    t0 = np.deg2rad(5)
    dth = np.angle(np.exp(1j * (th - t0)))
    bulge = jagged((rho < R + 0.17 * R * np.clip(1 - (dth / 0.28) ** 2, 0, 1) ** 0.5) & (np.abs(dth) < 0.28) & ~filled | False, 2)
    bulge &= ~filled
    src = polar_sample(img, cx, cy, np.clip(2 * R - rho - 0.04 * R, r, R), th + 0.5)
    out = blend(img, src * 0.9, bulge, 1.2)
    out = rim_shade(out, bulge | filled, 3, 0.6)
    return out, "lump_on_rim"


def flash(k):
    img, m, filled, bore, cx, cy, R, r, xx, yy = load(k)
    rho, th = np.hypot(xx - cx, yy - cy), np.arctan2(yy - cy, xx - cx)
    t0 = np.deg2rad(150)
    dth = np.angle(np.exp(1j * (th - t0)))
    thick = 0.13 * R * np.clip(1 - (dth / 0.5) ** 2, 0, 1)
    fin = (rho > R - 2) & (rho < R + thick) & (np.abs(dth) < 0.5) & ~filled
    src = polar_sample(img, cx, cy, np.full_like(rho, R * 0.96), th)
    out = rim_shade(blend(img, np.clip(src * 1.15 + 15, 0, 255), fin, 0.8), fin | filled, 2, 0.75)
    return out, "flash_on_rim"


JOBS = [("02", chip), ("02", through_hole), ("02", blocked_bore), ("02", porosity),
        ("09", through_crack), ("09", lump), ("09", flash), ("10", through_hole), ("14", chip)]
for k, fn in JOBS:
    out, name = fn(k)
    noise = rng.normal(0, 1.5, out.shape)
    Image.fromarray(np.clip(out + noise, 0, 255).astype(np.uint8)).save(G / f"defect_{name}_{k}.jpg", quality=92)
    print("wrote", f"defect_{name}_{k}.jpg")

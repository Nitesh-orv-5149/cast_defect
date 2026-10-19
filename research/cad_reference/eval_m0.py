"""M0 evaluation: CAD-only inspection (no good photos) on the synthetic photo set.

usage: python eval_m0.py [--recompute]
Measures every photo once (cached in data/m0_stats.json), prints per-group distributions of the decision
statistics, then verdict tables for increasing sets of checks. Writes results_m0.txt.
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cadref import build_profile, decide, load_mesh, measure

HERE = Path(__file__).parent
PHOTOS = HERE / "data" / "photos"
CACHE = HERE / "data" / "m0_stats.json"


def camera_finish(img, seed):
    """Sensor noise, slight defocus and JPEG, deterministic per photo."""
    r = random.Random(seed)
    img = cv2.GaussianBlur(img, (0, 0), r.uniform(0.3, 0.9))
    img = np.clip(img + np.random.default_rng(seed).normal(0, r.uniform(2, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, r.randint(70, 92)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def group(j):
    if j["set"] == "handheld":
        return "handheld/" + j["kind"] + ("/" + (j.get("defect") or j.get("other")) if j["kind"] != "good" else "")
    if j["kind"] == "good":
        return "good/" + ("stained" if j["stain"] else "clean")
    return f"{j['kind']}/{j.get('defect') or j.get('other')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    labels = json.load(open(PHOTOS / "labels.json"))
    if CACHE.exists() and not args.recompute:
        stats = json.load(open(CACHE))
    else:
        t = time.time()
        prof = build_profile(*load_mesh(HERE / "data" / "parts" / "ref.stl"))
        print(f"profile: {len(prof.views)} template views in {time.time() - t:.1f}s", flush=True)
        stats = {}
        for i, j in enumerate(labels):
            img = camera_finish(cv2.imread(str(PHOTOS / f"{j['id']}.png")), j["seed"])
            t = time.time()
            st = measure(prof, img)
            st["ms"] = (time.time() - t) * 1000
            stats[j["id"]] = st
            if i % 50 == 0:
                print(f"{i}/{len(labels)}", flush=True)
        json.dump(stats, open(CACHE, "w"))

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    by = collections.defaultdict(list)
    for j in labels:
        by[group(j)].append((j, stats[j["id"]]))

    out("Decision statistics per group (min / median / max)")
    out(f"{'group':36s} {'n':>3s}  {'IoU':^20s}  {'feature ratio':^17s}  {'weakest chunk rel':^17s}  {'longest stray mm (lvl1)':^22s}")
    for g in sorted(by):
        xs = [s for _, s in by[g] if not s.get("no_part")]
        iou = [s["iou"] for s in xs]
        fr = [s["feature_ratio"] for s in xs]
        wc = [min(c["support"] for c in s["chunks"]) / np.median([c["support"] for c in s["chunks"]]) for s in xs if s["chunks"]]
        sl = [max([L for _, L in s["stray"][1]] or [0]) for s in xs]
        f3 = lambda v, p=3: "/".join(f"{x:.{p}f}" for x in (min(v), np.median(v), max(v))) if v else "-"
        out(f"{g:36s} {len(by[g]):3d}  {f3(iou):20s}  {f3(fr, 2):17s}  {f3(wc, 2):17s}  {f3(sl, 1):22s}")

    want = {"good": "PASS", "defect": "REJECT", "other": "WRONG PART"}
    configs = [
        ("A. geometry only (wrong part, holes, outline)", {"use_features": False, "use_stray": False}),
        ("B. A + raised-feature check", {"use_features": True, "use_stray": False}),
        ("C. B + stray-edge check (cracks, pits)", {"use_features": True, "use_stray": True}),
    ]
    for name, cfg in configs:
        out(f"\n{name}")
        out(f"{'group':36s} {'n':>3s}  {'correct':>7s}  PASS REJECT WRONG  reasons")
        for g in sorted(by):
            items = [(j, *decide(s, cfg)) for j, s in by[g]]
            c = collections.Counter(v for _, v, _ in items)
            right = sum(v == want[j["kind"]] for j, v, _ in items) / len(items)
            reasons = collections.Counter(f["type"] for *_, F in items for f in F)
            out(f"{g:36s} {len(items):3d}  {right:7.0%}  {c['PASS']:4d} {c['REJECT']:6d} {c['WRONG PART']:5d}  {dict(reasons) if reasons else ''}")
        goods = [decide(s, cfg)[1] for j, s in stats_items(labels, stats) if j["kind"] == "good"]
        alarms = sum(any(f["type"] in ("blocked_hole", "unexpected_hole") for f in F) for F in goods)
        out(f"good parts with a designed hole flagged: {alarms}/{len(goods)}")
    ms = [s["ms"] for s in stats.values() if "ms" in s]
    out(f"\ntime per photo: median {np.median(ms):.0f} ms, max {max(ms):.0f} ms (Python + OpenCV, 1 core)")
    (HERE / "results_m0.txt").write_text("\n".join(lines) + "\n")


def stats_items(labels, stats):
    return [(j, stats[j["id"]]) for j in labels]


if __name__ == "__main__":
    sys.exit(main())

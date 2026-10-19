"""Generalisation test: run the M0 geometry checks on other castings (asymmetric L-bracket, tall pump housing).

usage: python eval_general.py [--recompute] [--perspective]
Same pipeline and default thresholds as the impeller; only the CAD file changes.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cadref import build_profile, decide, load_mesh, measure
from eval_m0 import camera_finish

HERE = Path(__file__).parent
PHOTOS = HERE / "data" / "photos_general"


def group(j):
    if j["set"] == "handheld":
        return "handheld/good"
    if j["kind"] == "good":
        return "good/" + ("stained" if j["stain"] else "clean")
    return f"{j['kind']}/{j['defect'] or j['other']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--perspective", type=float, default=0.0, help="camera distance (mm) for perspective templates; 0 = orthographic")
    args = ap.parse_args()
    labels = json.load(open(PHOTOS / "labels.json"))
    cache = HERE / "data" / f"general_stats{'_persp' if args.perspective else ''}.json"
    if cache.exists() and not args.recompute:
        stats = json.load(open(cache))
    else:
        stats = {}
        for fam in sorted({j["family"] for j in labels}):
            t = time.time()
            kw = {"cam_dist": args.perspective} if args.perspective else {}
            prof = build_profile(*load_mesh(HERE / "data" / "parts_general" / fam / "ref.stl"), **kw)
            print(f"{fam}: profile {len(prof.views)} views in {time.time() - t:.1f}s", flush=True)
            for j in [j for j in labels if j["family"] == fam]:
                img = camera_finish(cv2.imread(str(PHOTOS / f"{j['id']}.png")), j["seed"])
                t = time.time()
                st = measure(prof, img)
                st["ms"] = (time.time() - t) * 1000
                stats[j["id"]] = st
        json.dump(stats, open(cache, "w"))
    lines = []
    def out(s=""):
        print(s); lines.append(s)
    want = {"good": "PASS", "defect": "REJECT", "other": "WRONG PART"}
    cfg = {"use_features": False, "use_stray": False}
    for fam in sorted({j["family"] for j in labels}):
        out(f"\n=== {fam} ({'perspective' if args.perspective else 'orthographic'} templates), geometry checks, default thresholds")
        out(f"{'group':34s} {'n':>3s}  {'correct':>7s}  PASS REJECT WRONG   IoU min/med      feature ratio min/med   reasons")
        by = collections.defaultdict(list)
        for j in labels:
            if j["family"] == fam:
                by[group(j)].append((j, stats[j["id"]]))
        for g in sorted(by):
            items = [(j, s, *decide(s, cfg)) for j, s in by[g]]
            c = collections.Counter(v for *_, v, _ in items)
            right = sum(v == want[j["kind"]] for j, _, v, _ in items) / len(items)
            iou = [s["iou"] for _, s, *_ in items if "iou" in s]
            fr = [s["feature_ratio"] for _, s, *_ in items if "iou" in s]
            reasons = collections.Counter(f["type"] for *_, F in items for f in F)
            out(f"{g:34s} {len(items):3d}  {right:7.0%}  {c['PASS']:4d} {c['REJECT']:6d} {c['WRONG PART']:5d}   "
                f"{min(iou):.3f}/{np.median(iou):.3f}     {min(fr):5.2f}/{np.median(fr):5.2f}           {dict(reasons) if reasons else ''}")
        goods = [decide(s, cfg)[1] for j, s in ((j, stats[j["id"]]) for j in labels if j["family"] == fam) if j["kind"] == "good"]
        out(f"good parts with a designed hole flagged: {sum(any(f['type'] in ('blocked_hole', 'unexpected_hole') for f in F) for F in goods)}/{len(goods)}")
    (HERE / f"results_general{'_persp' if args.perspective else ''}.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

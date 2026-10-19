"""Run the CAD-reference inspector on a folder of photos and grade it against answers.json if present.

usage: python run_testkit.py [--cad testkit/conrod.stl] [--photos testkit/photos] [--camera-mm 322] [--out testkit/results]
Thresholds are the defaults in cadref.DEFAULTS (geometry checks only), unchanged for this part.
--camera-mm is the station's camera height above the table (used for perspective templates).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from cadref import align, build_profile, decide, from_template, load_mesh, measure, segment

HERE = Path(__file__).parent
WANT = {"good": "PASS", "defect": "REJECT", "other": "WRONG PART"}


def overlay(prof, img, verdict, findings, label):
    al = align(prof, segment(img))
    rd = prof.views[al["view"]][2]
    e = from_template(rd["edges"].astype(np.uint8) * 255, al["H"], img.shape) > 0
    hb = from_template(rd["holes"].astype(np.uint8) * 255, al["H"], img.shape) > 0
    o = img.copy()
    o[e] = (60, 60, 255)
    o[hb] = (o[hb] * 0.4 + np.array([255, 170, 40]) * 0.6).astype(np.uint8)
    col = {"PASS": (60, 170, 60), "REJECT": (50, 50, 220), "WRONG PART": (0, 150, 230)}.get(verdict, (200, 200, 200))
    cv2.rectangle(o, (0, 0), (o.shape[1], 58), (25, 25, 25), -1)
    cv2.putText(o, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    why = ", ".join(sorted({f["type"].replace("_", " ") for f in findings if f["type"] != "mismatch"}))
    cv2.putText(o, f"{verdict}  {why}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad", default=str(HERE / "testkit" / "conrod.stl"))
    ap.add_argument("--photos", default=str(HERE / "testkit" / "photos"))
    ap.add_argument("--camera-mm", type=float, default=322.0)
    ap.add_argument("--out", default=str(HERE / "testkit" / "results"))
    args = ap.parse_args()
    prof = build_profile(*load_mesh(args.cad), cam_dist=args.camera_mm)
    answers_path = Path(args.photos).parent / "answers.json"
    answers = json.load(open(answers_path)) if answers_path.exists() else {}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, right = [], 0
    for p in sorted(Path(args.photos).glob("*.jpg")):
        img = cv2.imread(str(p))
        st = measure(prof, img)
        verdict, findings = decide(st, {"use_features": False, "use_stray": False})
        a = answers.get(p.stem)
        ok = a is not None and verdict == WANT[a["kind"]]
        right += ok
        truth = f"truth: {a['truth']}" if a else ""
        rows.append(f"{p.stem}  {verdict:10s}  {'OK ' if ok else 'MISS' if a else '':4s}  {truth}")
        print(rows[-1], flush=True)
        cv2.imwrite(str(out / f"{p.stem}.jpg"), overlay(prof, img, verdict, findings, f"{p.stem}  {truth}"))
    if answers:
        rows.append(f"\n{right}/{len(rows)} correct")
        print(rows[-1])
    (out / "verdicts.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()

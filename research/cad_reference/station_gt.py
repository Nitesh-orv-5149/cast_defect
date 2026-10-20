"""Ground-truth defect location for the reference-images station set: each defect photo minus its defect-free twin
(same pose, light and texture seed), as 28x28 cells over the app's 448 px crop. Writes js/station_gt.json.

usage: python station_gt.py   (after rendering data/photos_station and data/photos_station_twin)
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent


def main():
    gt = {}
    for j in json.load(open(HERE / "data/photos_station/labels.json")):
        if j["kind"] == "good":
            continue
        a = cv2.imread(str(HERE / f"data/photos_station/{j['id']}.png")).astype(np.float32)
        b = cv2.imread(str(HERE / f"data/photos_station_twin/{j['id']}_twin.png")).astype(np.float32)
        m = (cv2.GaussianBlur(np.abs(a - b).max(2), (0, 0), 1.5) > 18).astype(np.uint8)
        cells = cv2.resize(cv2.resize(m, (448, 448), interpolation=cv2.INTER_NEAREST).astype(np.float32), (28, 28), interpolation=cv2.INTER_AREA)
        gt[j["id"]] = {"cells": (cells > 0.02).astype(int).flatten().tolist(), "px": int(m.sum())}
    json.dump(gt, open(HERE / "js/station_gt.json", "w"))
    print(len(gt), "defect photos")


if __name__ == "__main__":
    main()

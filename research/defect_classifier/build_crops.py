"""Close-up defect crops for the tab 1 defect-type classifier.

Each labelled defect box in the Roboflow casting sets becomes one square crop (box side x MARGIN, centred on the
box), which is what a phone close-up of that defect looks like. "good" crops are cut from defect-free impellers
(Kaggle ok_front) at the same sizes, plus real sand-cast surfaces from SIFA. Train/val/test are split by source
photo (a hash of the file name with Roboflow's augmentation suffix removed), so augmented copies and photos that
appear in two Roboflow projects never land on both sides.

usage: python build_crops.py <datasets dir> <out dir>
  <datasets dir>/roboflow/<project>/{train,valid,test}/{images,labels}   (YOLO export)
  <datasets dir>/kaggle/impeller/casting_512x512/casting_512x512/ok_front/*.jpeg
  <datasets dir>/kaggle/sifa/.../Real Images Labeled/{images,labels}
"""
import csv
import glob
import hashlib
import math
import os
import random
import re
import sys
from collections import Counter

import yaml
import numpy as np
from PIL import Image

MARGIN, MIN_SIDE, OUT_SIZE = 1.6, 12, 192
PROJECTS = ["casing_defect_detection", "casting_dataset_kaggle", "casting-defects-onf6n-xt28x", "casting-dataset-2"]
# raw_data_casting is left out: its boxes do not line up with the defects
CLASS = {"BlowHoles": "gas", "Blowholes": "gas", "blow hole": "gas", "Porosity": "gas",
         "Shrinkage": "shrinkage", "crack": "crack", "Misrun": "misrun", "Cold shot": "misrun",
         "Burr": "flash_burr", "Flash": "flash_burr", "Plash": "flash_burr",
         "MouldDefect": "mould_surface", "Scar": "mould_surface", "Scratch": "scratch", "scratch": "scratch"}


def split_of(key):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def base_name(path):
    return re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", os.path.basename(path)).lower()


def square_crop(im, cx, cy, side):
    """Square crop centred on (cx, cy), padded by edge replication where it leaves the photo."""
    side = max(int(round(side)), MIN_SIDE)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = max(0, -x0, -y0, x0 + side - im.width, y0 + side - im.height)
    if pad:
        im = _pad_edge(im, pad)
        x0, y0 = x0 + pad, y0 + pad
    return im.crop((x0, y0, x0 + side, y0 + side)).resize((OUT_SIZE, OUT_SIZE), Image.BICUBIC)


def _pad_edge(im, pad):
    a = np.asarray(im)
    a = np.pad(a, ((pad, pad), (pad, pad)) + ((0, 0),) * (a.ndim - 2), mode="edge")
    return Image.fromarray(a)


def main():
    root, out = sys.argv[1], sys.argv[2]
    rows, sizes, seen_photo = [], [], set()
    for proj in PROJECTS:
        d = f"{root}/roboflow/{proj}"
        names = yaml.safe_load(open(f"{d}/data.yaml"))["names"]
        done = set()
        for lab in sorted(glob.glob(f"{d}/*/labels/*.txt")):
            key = base_name(lab)
            if key in done:  # Roboflow augmentation copy of a photo already used from this project
                continue
            done.add(key)
            stem = lab.replace("/labels/", "/images/").rsplit(".", 1)[0]
            img = next((stem + e for e in (".jpg", ".jpeg", ".png") if os.path.exists(stem + e)), None)
            if not img:
                continue
            im = Image.open(img).convert("RGB")
            W, H = im.size
            for i, line in enumerate(open(lab)):
                p = line.split()
                if len(p) < 5 or names[int(p[0])] not in CLASS:
                    continue
                cls = CLASS[names[int(p[0])]]
                x, y, w, h = float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
                if max(w, h) < 6:
                    continue
                side = max(w, h) * MARGIN
                sizes.append(side / W)
                rows.append((cls, key, f"{proj}:{os.path.basename(img)}#{i}", square_crop(im, x, y, side)))
            seen_photo.add(key)
    # good surface: same crop sizes, on defect-free impellers (inside the part) and on SIFA sand castings
    rnd = random.Random(0)
    oks = sorted(glob.glob(f"{root}/kaggle/impeller/casting_512x512/casting_512x512/ok_front/*.jpeg"))
    for f in oks:
        im = Image.open(f).convert("RGB")
        W = im.width
        for k in range(4):
            side = rnd.choice(sizes) * W
            a, r = rnd.uniform(0, 6.283), rnd.uniform(0.05, 0.42) * W
            rows.append(("good", "ok:" + os.path.basename(f), f"ok:{os.path.basename(f)}#{k}",
                         square_crop(im, W / 2 + r * math.cos(a), W / 2 + r * math.sin(a), side)))
    sifa = glob.glob(f"{root}/kaggle/sifa/**/Real Images Labeled/labels/*.txt", recursive=True)
    for lab in sorted(sifa)[:400]:
        img = lab.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
        if not os.path.exists(img):
            continue
        im = Image.open(img).convert("RGB")
        W, H = im.size
        for line in open(lab):
            p = line.split()
            if len(p) >= 5 and int(p[0]) == 2:  # "part"
                x, y, w, h = float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
                side = min(w, h) * rnd.uniform(0.15, 0.35)
                rows.append(("good", "sifa:" + base_name(lab), f"sifa:{os.path.basename(img)}",
                             square_crop(im, x + rnd.uniform(-0.25, 0.25) * w, y + rnd.uniform(-0.25, 0.25) * h, side)))
                break
    counts = Counter()
    with open(f"{out}/manifest.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "class", "photo", "source", "file"])
        for n, (cls, key, src, crop) in enumerate(rows):
            sp = split_of(key)
            os.makedirs(f"{out}/{sp}/{cls}", exist_ok=True)
            fn = f"{sp}/{cls}/{n:06d}.png"
            crop.save(f"{out}/{fn}")
            wr.writerow([sp, cls, key, src, fn])
            counts[(sp, cls)] += 1
    for cls in sorted({c for _, c in counts}):
        print(f"{cls:14s} " + "  ".join(f"{sp} {counts[(sp, cls)]:5d}" for sp in ("train", "val", "test")))


if __name__ == "__main__":
    main()

"""Reference-based (PatchCore-style) anomaly detection on the Kaggle casting set.
Enrol N good parts -> patch memory bank -> score test parts by nearest-neighbour patch distance.
Backbones are ImageNet-pretrained only (no casting training). Usage: python ref_eval.py <project_dir>
"""
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(sys.argv[1]) / "data" / "casting_data"
OUT = Path(__file__).parent
torch.set_grad_enabled(False)
tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def mobilenet():
    f = models.mobilenet_v2(weights="IMAGENET1K_V1").features.eval()
    def run(x):
        a = f[:7](x); b = f[7:14](a)            # stride 8 (32ch, 28x28), stride 16 (96ch, 14x14)
        return a, b
    return run


def resnet18():
    m = models.resnet18(weights="IMAGENET1K_V1").eval()
    def run(x):
        x = m.maxpool(m.relu(m.bn1(m.conv1(x)))); x = m.layer1(x); a = m.layer2(x); b = m.layer3(a)
        return a, b
    return run


def patches(run, x):
    a, b = run(x)
    a = F.avg_pool2d(a, 3, 1, 1); b = F.avg_pool2d(b, 3, 1, 1)          # local neighbourhood context
    b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    p = torch.cat([a, b], 1)                                            # N, C, 28, 28
    return p.permute(0, 2, 3, 1).reshape(x.shape[0], -1, p.shape[1])   # N, 784, C


def load(paths):
    return torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])


def score(bank, q):  # q: (784, C) -> per-patch NN distance
    return torch.cdist(q, bank).min(1).values


def auroc(neg, pos):
    s = np.r_[neg, pos]; y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    random.seed(0); torch.manual_seed(0)
    ok_train = sorted((ROOT / "train/ok_front").glob("*.jpeg"))
    test_ok = sorted((ROOT / "test/ok_front").glob("*.jpeg"))
    test_def = sorted((ROOT / "test/def_front").glob("*.jpeg"))
    lines = []
    for bname, bfn in [("MobileNetV2", mobilenet), ("ResNet18", resnet18)]:
        run = bfn()
        tq = {}
        t0 = time.time()
        for k, ps in [("ok", test_ok), ("def", test_def)]:
            tq[k] = torch.cat([patches(run, load(ps[i:i + 32])) for i in range(0, len(ps), 32)])
        print(f"{bname}: test features in {time.time() - t0:.0f}s, patch dim {tq['ok'].shape[-1]}")
        for shots in (1, 5, 10):
            for rot in (False, True):
                refs = load(random.sample(ok_train, shots))
                if rot:  # augment references with 90-degree rotations and a flip
                    refs = torch.cat([torch.rot90(refs, k, (2, 3)) for k in range(4)] + [refs.flip(3)])
                bank = patches(run, refs).reshape(-1, tq["ok"].shape[-1])
                # auto threshold from references only: leave-one-image-out max score
                per = patches(run, refs)
                loo = [score(torch.cat([per[:i], per[i + 1:]]).reshape(-1, per.shape[-1]), per[i]).max().item()
                       for i in range(len(per))] if len(per) > 1 else [float("nan")]
                t1 = time.time()
                s_ok = np.array([score(bank, q).max().item() for q in tq["ok"]])
                s_def = np.array([score(bank, q).max().item() for q in tq["def"]])
                ms = (time.time() - t1) / (len(s_ok) + len(s_def)) * 1000
                auc = auroc(s_ok, s_def)
                thr = max(loo) * 1.05
                rec, far = (s_def > thr).mean(), (s_ok > thr).mean()
                t95 = np.quantile(s_def, 0.05); far95 = (s_ok > t95).mean()
                line = (f"{bname:11s} shots={shots:2d} rot={'Y' if rot else 'N'} bank={bank.shape[0]:6d}  AUROC={auc:.3f}  "
                        f"auto-thr: catch {rec:.0%} def, false-reject {far:.0%} ok | @95% catch: false-reject {far95:.0%}  "
                        f"({ms:.0f} ms/img CPU)")
                print(line); lines.append(line)
    (OUT / "results.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

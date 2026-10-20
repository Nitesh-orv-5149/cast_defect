"""Train the tab 1 close-up defect-type classifier on the crops from build_crops.py and export it for the browser.

EfficientNet-B0 (ImageNet weights from the Hugging Face hub via timm), grayscale input so the phone sees what the
grayscale training photos show, class-weighted loss for the rare classes. Picks the epoch with the best validation
macro-F1, reports the held-out test set, then exports web/types.onnx (outputs: probs, features) and web/types.json
(classes, input size, normalisation, classifier weights for the heat map).

usage: python train_types.py <crops dir> [--epochs 14] [--size 160] [--out ../../web]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Export(nn.Module):
    """(probs, final feature maps); with GAP + Linear, the heat map for class c is ReLU(sum_k w[c,k] A_k)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        a = self.m.forward_features(x)
        return F.softmax(self.m.classifier(a.mean((2, 3))), 1), a


def report(model, dl, classes, title):
    model.eval()
    cm = np.zeros((len(classes), len(classes)), int)
    with torch.no_grad():
        for x, y in dl:
            for t, p in zip(y.numpy(), model(x).argmax(1).numpy()):
                cm[t, p] += 1
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    lines = [f"{title}: accuracy {cm.trace() / cm.sum():.3f}  macro-F1 {f1.mean():.3f}"]
    lines += [f"  {c:14s} n={cm[i].sum():4d}  recall {rec[i]:.2f}  precision {prec[i]:.2f}" for i, c in enumerate(classes)]
    lines += ["  confusion (rows = truth): " + " ".join(f"{c[:5]:>6s}" for c in classes)]
    lines += [f"  {c[:12]:>12s} " + " ".join(f"{v:6d}" for v in cm[i]) for i, c in enumerate(classes)]
    return f1.mean(), "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("crops")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "web"))
    a = ap.parse_args()
    torch.manual_seed(0)
    norm = transforms.Normalize(MEAN, STD)
    tf_train = transforms.Compose([
        transforms.Grayscale(3),
        transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(a.size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.35, 0.35),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.3, 1.5))], p=0.3),
        transforms.ToTensor(), norm])
    tf_eval = transforms.Compose([transforms.Grayscale(3), transforms.Resize(a.size), transforms.CenterCrop(a.size), transforms.ToTensor(), norm])
    sets = {s: datasets.ImageFolder(Path(a.crops) / s, tf_train if s == "train" else tf_eval) for s in ("train", "val", "test")}
    classes = sets["train"].classes
    assert all(sets[s].classes == classes for s in sets)
    n = Counter(sets["train"].targets)
    weight = torch.tensor([(max(n.values()) / n[i]) ** 0.5 for i in range(len(classes))], dtype=torch.float)
    print("classes:", classes, "| train counts:", [n[i] for i in range(len(classes))], "| loss weights:", weight.numpy().round(2))
    dl = torch.utils.data.DataLoader(sets["train"], 48, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    vl = torch.utils.data.DataLoader(sets["val"], 96, num_workers=2)
    tl = torch.utils.data.DataLoader(sets["test"], 96, num_workers=2)

    model = timm.create_model("efficientnet_b0.ra_in1k", pretrained=True, num_classes=len(classes), drop_rate=0.2)
    opt = torch.optim.AdamW(model.parameters(), 6e-4, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 6e-4, total_steps=a.epochs * len(dl), pct_start=0.15)
    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, weight=weight, label_smoothing=0.1)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        f1, txt = report(model, vl, classes, "val")
        print(f"epoch {ep + 1}/{a.epochs}  loss {tot / len(dl):.3f}  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(report(model, vl, classes, "best val")[1])
    print(report(model, tl, classes, "TEST (held-out photos)")[1])

    out = Path(a.out)
    exp = Export(model).eval()
    x = torch.zeros(1, 3, a.size, a.size)
    torch.onnx.export(exp, x, out / "types.onnx", input_names=["input"], output_names=["probs", "features"], opset_version=17, dynamo=False)
    json.dump({"size": a.size, "mean": MEAN, "std": STD, "grayscale": True, "classes": classes,
               "cam_weights": model.classifier.weight.detach().numpy().round(5).tolist()}, open(out / "types.json", "w"))
    import onnxruntime as ort
    xs = torch.stack([sets["test"][i][0] for i in range(0, len(sets["test"]), max(1, len(sets["test"]) // 8))])
    for xi in xs:
        p, f = ort.InferenceSession(str(out / "types.onnx")).run(None, {"input": xi[None].numpy()})
        with torch.no_grad():
            tp, tf = exp(xi[None])
        assert np.allclose(p, tp.numpy(), atol=1e-4) and np.allclose(f, tf.numpy(), atol=1e-3), "ONNX != torch"
    print(f"exported {out / 'types.onnx'} ({(out / 'types.onnx').stat().st_size / 1e6:.1f} MB), ONNX matches torch")


if __name__ == "__main__":
    main()

"""Fine-tune MobileNetV2 on the Kaggle casting dataset, export web/model.onnx + web/meta.json.

Expects data/casting_data/{train,test}/{def_front,ok_front}/*.jpeg
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

DATA = Path("data/casting_data")
OUT = Path("web")
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EPOCHS = 4


class Export(nn.Module):
    """Outputs (probs, last feature maps). With GAP+Linear head, Grad-CAM = ReLU(sum_k w[c,k] * A_k)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        a = self.m.features(x)
        return F.softmax(self.m.classifier(a.mean((2, 3))), 1), a


def gradcam_autograd(model, x, c):
    a = model.features(x)
    a.retain_grad()
    model.classifier(a.mean((2, 3)))[0, c].backward()
    return F.relu((a.grad.mean((2, 3))[:, :, None, None] * a).sum(1))[0].detach()


def cam_from_weights(a, w):
    return F.relu(torch.einsum("k,khw->hw", w, a[0]))


def main():
    if not (DATA / "train").is_dir():
        sys.exit(f"Dataset missing: unzip Kaggle 'casting product image data' into {DATA}/")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = transforms.Normalize(MEAN, STD)
    tf_train = transforms.Compose([
        transforms.Resize(SIZE), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(0.3, 0.3), transforms.ToTensor(), norm])
    tf_test = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor(), norm])
    train = datasets.ImageFolder(DATA / "train", tf_train)
    test = datasets.ImageFolder(DATA / "test", tf_test)
    defect = train.class_to_idx["def_front"]
    dl = torch.utils.data.DataLoader(train, 32, shuffle=True, num_workers=2)
    tl = torch.utils.data.DataLoader(test, 64, num_workers=2)

    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(1280, 2)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), 3e-4)
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            opt.zero_grad()
            F.cross_entropy(model(x.to(dev)), y.to(dev)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            correct = sum((model(x.to(dev)).argmax(1).cpu() == y).sum().item() for x, y in tl)
        print(f"epoch {ep + 1}: test acc {correct / len(test):.4f}")

    model.cpu().eval()
    x = test[0][0][None]
    w = model.classifier[1].weight[defect].detach()
    # Self-check: weight-based CAM (what the phone computes) == autograd Grad-CAM
    a = model.features(x).detach()
    g, c = gradcam_autograd(model, x.clone(), defect).flatten(), cam_from_weights(a, w).flatten()
    cos = F.cosine_similarity(g, c, 0).item() if g.norm() > 0 and c.norm() > 0 else float(torch.equal(g, c))
    assert cos > 0.99, f"CAM mismatch, cosine={cos}"
    print(f"Grad-CAM equivalence check: cosine={cos:.5f}")

    OUT.mkdir(exist_ok=True)
    exp = Export(model).eval()  # a fresh wrapper defaults to train mode -> BN would export with batch stats
    torch.onnx.export(exp,torch.zeros(1, 3, SIZE, SIZE), OUT / "model.onnx",
                      input_names=["input"], output_names=["probs", "features"], opset_version=17, dynamo=False)
    json.dump({"size": SIZE, "mean": MEAN, "std": STD, "classes": train.classes, "defect_idx": defect,
               "cam_weights": w.tolist()}, open(OUT / "meta.json", "w"))

    # Self-check: exported ONNX matches torch
    import onnxruntime as ort
    p, f = ort.InferenceSession(str(OUT / "model.onnx")).run(None, {"input": x.numpy()})
    with torch.no_grad():
        tp, tf = exp(x)
    assert np.allclose(p, tp.numpy(), atol=1e-4) and np.allclose(f, tf.numpy(), atol=1e-3), "ONNX != torch"
    print(f"Exported {OUT / 'model.onnx'} ({(OUT / 'model.onnx').stat().st_size / 1e6:.1f} MB), ONNX check OK")


if __name__ == "__main__":
    main()

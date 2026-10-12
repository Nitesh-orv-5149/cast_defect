"""Export the reference-mode (PatchCore-style) models for the web app. No training: ImageNet ResNet18 only.

web/ref_backbone.onnx : image (1,3,224,224) -> patch features (784, C), one row per 8x8-pixel patch
web/ref_nn.onnx       : query (N,C), bank (M,C) -> distance of each query row to its nearest bank row (N)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

OUT = Path("web")
PROJ = 128  # random projection 384 -> PROJ dims keeps the phone fast (set None to keep all 384)


class Patches(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights="IMAGENET1K_V1").eval()
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3
        g = torch.Generator().manual_seed(123)  # same projection as research/reference_anomaly/deploy_eval.py
        self.register_buffer("R", torch.randn(384, PROJ, generator=g) / PROJ ** 0.5 if PROJ else torch.eye(384))

    def forward(self, x):
        a = self.l2(self.stem(x)); b = self.l3(a)
        a = F.avg_pool2d(a, 3, 1, 1); b = F.avg_pool2d(b, 3, 1, 1)
        b = F.interpolate(b, size=(28, 28), mode="bilinear", align_corners=False)
        p = torch.cat([a, b], 1)[0].reshape(384, 784).t()  # (784, 384)
        return p @ self.R


class NearestDist(nn.Module):
    def forward(self, q, b):
        d = (q * q).sum(1, keepdim=True) - 2 * q @ b.t() + (b * b).sum(1)[None]
        return d.min(1).values.clamp(min=0).sqrt()


def main():
    OUT.mkdir(exist_ok=True)
    pm = Patches().eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(pm, x, OUT / "ref_backbone.onnx", input_names=["input"], output_names=["patches"],
                          opset_version=17, dynamo=False)
        q, b = torch.randn(784, pm.R.shape[1]), torch.randn(5000, pm.R.shape[1])
        torch.onnx.export(NearestDist(), (q, b), OUT / "ref_nn.onnx", input_names=["q", "b"], output_names=["d"],
                          dynamic_axes={"q": {0: "n"}, "b": {0: "m"}, "d": {0: "n"}}, opset_version=17, dynamo=False)

    # Self-check: ONNX matches torch, and the nearest-distance graph matches cdist
    import onnxruntime as ort
    p = ort.InferenceSession(str(OUT / "ref_backbone.onnx")).run(None, {"input": x.numpy()})[0]
    with torch.no_grad():
        tp = pm(x).numpy()
    assert np.allclose(p, tp, atol=1e-3), "backbone ONNX != torch"
    d = ort.InferenceSession(str(OUT / "ref_nn.onnx")).run(None, {"q": q.numpy(), "b": b[:1234].numpy()})[0]
    assert np.allclose(d, torch.cdist(q, b[:1234]).min(1).values.numpy(), atol=1e-3), "nn ONNX != cdist"
    mb = (OUT / "ref_backbone.onnx").stat().st_size / 1e6
    print(f"Exported ref_backbone.onnx ({mb:.1f} MB, {p.shape}) and ref_nn.onnx; checks OK")


if __name__ == "__main__":
    sys.exit(main())

"""Round 2: more shots, 8 rotation angles, subsampled bank, top-k patch scoring, heatmap samples."""
import random, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
sys.path.insert(0, str(Path(__file__).parent))
from ref_eval import ROOT, OUT, load, patches, resnet18, mobilenet, auroc

torch.set_grad_enabled(False)
def rots(x, n):  # n evenly spaced rotations + horizontal flip of each
    out = [TF.rotate(x, 360 * k / n) for k in range(n)]
    return torch.cat(out + [o.flip(3) for o in out])

def evaluate(name, run, shots, n_rot, bank_cap, topk, tq, ok_train):
    random.seed(1); torch.manual_seed(1)
    refs = load(random.sample(ok_train, shots))
    bank = torch.cat([patches(run, rots(refs[i:i+1], n_rot)).reshape(-1, tq["ok"].shape[-1]) for i in range(shots)])
    if bank.shape[0] > bank_cap: bank = bank[torch.randperm(bank.shape[0])[:bank_cap]]
    def s(q): return torch.cdist(q, bank).min(1).values.topk(topk).values.mean().item()
    t = time.time()
    so = np.array([s(q) for q in tq["ok"]]); sd = np.array([s(q) for q in tq["def"]])
    ms = (time.time() - t) / (len(so) + len(sd)) * 1000
    t95 = np.quantile(sd, 0.05); t90 = np.quantile(sd, 0.10)
    return (f"{name:11s} shots={shots:3d} rot={n_rot} bank={bank.shape[0]:6d} top{topk:<3d} AUROC={auroc(so, sd):.3f}  "
            f"@95% catch false-reject {(so > t95).mean():.0%} | @90% catch false-reject {(so > t90).mean():.0%}  ({ms:.0f} ms/img)")

ok_train = sorted((ROOT / "train/ok_front").glob("*.jpeg"))
test_ok = sorted((ROOT / "test/ok_front").glob("*.jpeg")); test_def = sorted((ROOT / "test/def_front").glob("*.jpeg"))
lines = []
for name, fn in [("ResNet18", resnet18), ("MobileNetV2", mobilenet)]:
    run = fn(); tq = {k: torch.cat([patches(run, load(ps[i:i+32])) for i in range(0, len(ps), 32)]) for k, ps in [("ok", test_ok), ("def", test_def)]}
    for shots, n_rot, topk in [(10, 8, 1), (10, 8, 10), (25, 8, 10), (50, 8, 10), (50, 4, 10)]:
        l = evaluate(name, run, shots, n_rot, 60000, topk, tq, ok_train); print(l, flush=True); lines.append(l)
(OUT / "results2.txt").write_text("\n".join(lines))

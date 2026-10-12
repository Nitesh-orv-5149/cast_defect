"""Round 3: user-chosen 1..20 references. Heavy augmentation of each reference, auto threshold from
held-out augmented views of the references only (no defect examples), 2 seeds."""
import random, sys, time
from pathlib import Path
import numpy as np, torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
sys.path.insert(0, str(Path(__file__).parent))
from ref_eval import ROOT, OUT, load, patches, resnet18, auroc

torch.set_grad_enabled(False)
jit = T.Compose([T.RandomAffine(180, translate=(0.04, 0.04), scale=(0.95, 1.05)), T.ColorJitter(0.15, 0.15)])
def bank_views(x):  # 8 rotations + flips
    out = [TF.rotate(x, 45 * k) for k in range(8)]; return torch.cat(out + [o.flip(3) for o in out])

ok_train = sorted((ROOT / "train/ok_front").glob("*.jpeg"))
rng = random.Random(0)
test_ok = rng.sample(sorted((ROOT / "test/ok_front").glob("*.jpeg")), 200)
test_def = rng.sample(sorted((ROOT / "test/def_front").glob("*.jpeg")), 200)
run = resnet18()
tq = {k: torch.cat([patches(run, load(ps[i:i+40])) for i in range(0, len(ps), 40)]) for k, ps in [("ok", test_ok), ("def", test_def)]}
C = tq["ok"].shape[-1]
lines = []
for shots in (1, 3, 5, 10, 20):
    for seed in (1, 2):
        random.seed(seed); torch.manual_seed(seed)
        refs = load(random.sample(ok_train, shots))
        bank = torch.cat([patches(run, bank_views(refs[i:i+1])).reshape(-1, C) for i in range(shots)])
        if bank.shape[0] > 60000: bank = bank[torch.randperm(bank.shape[0])[:60000]]
        sc = lambda q: torch.cdist(q, bank).min(1).values.topk(10).values.mean().item()
        val = torch.stack([jit(refs[i]) for i in range(shots) for _ in range(8)])   # unseen views of the good parts
        sv = np.array([sc(q) for q in patches(run, val)])
        thr = sv.max() * 1.05
        so = np.array([sc(q) for q in tq["ok"]]); sd = np.array([sc(q) for q in tq["def"]])
        l = (f"shots={shots:2d} seed={seed} bank={bank.shape[0]:6d} AUROC={auroc(so, sd):.3f}  AUTO threshold: catch {(sd > thr).mean():.0%} defects, "
             f"false-reject {(so > thr).mean():.0%} good | best-case @95% catch: false-reject {(so > np.quantile(sd, .05)).mean():.0%}")
        print(l, flush=True); lines.append(l)
(OUT / "results3.txt").write_text("\n".join(lines))

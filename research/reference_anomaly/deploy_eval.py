"""Which phone-sized config keeps accuracy? Configs differ in rotation views, bank cap and random projection."""
import random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from ref_eval import ROOT, load, patches, resnet18, auroc
torch.set_grad_enabled(False)
ok_train = sorted((ROOT / "train/ok_front").glob("*.jpeg"))
rng = random.Random(0)
test_ok = rng.sample(sorted((ROOT / "test/ok_front").glob("*.jpeg")), 200)
test_def = rng.sample(sorted((ROOT / "test/def_front").glob("*.jpeg")), 200)
run = resnet18()
tq = {k: torch.cat([patches(run, load(ps[i:i+40])) for i in range(0, len(ps), 40)]) for k, ps in [("ok", test_ok), ("def", test_def)]}
C = tq["ok"].shape[-1]
torch.manual_seed(123); R = torch.randn(C, 128) / 128 ** 0.5
def views(x, rot): return torch.cat([torch.rot90(x, k, (2, 3)) for k in range(4)]) if rot else x
for shots in (5, 10, 20):
    for rot, cap, proj in [(False, 10**9, False), (True, 10**9, False), (True, 30000, False), (True, 30000, True)]:
        res = []
        for seed in (1, 2):
            random.seed(seed); torch.manual_seed(seed)
            refs = load(random.sample(ok_train, shots))
            per = [patches(run, views(refs[i:i+1], rot)).reshape(-1, C) for i in range(shots)]
            P = (lambda t: t @ R) if proj else (lambda t: t)
            def bank_of(lst):
                b = torch.cat(lst); g = torch.Generator().manual_seed(0)
                return P(b[torch.randperm(b.shape[0], generator=g)[:cap]] if b.shape[0] > cap else b)
            bank = bank_of(per)
            sc = lambda q, b: torch.cdist(P(q), b).min(1).values.topk(10).values.mean().item()
            Q = patches(run, refs)
            cal = max(sc(Q[i], bank_of(per[:i] + per[i+1:])) for i in range(shots))
            so = np.array([sc(q, bank) for q in tq["ok"]]); sd = np.array([sc(q, bank) for q in tq["def"]])
            res.append((auroc(so, sd), *[((sd > cal*m).mean(), (so > cal*m).mean()) for m in (0.8, 0.9, 1.0)], bank.shape[0]))
        a = np.mean([r[0] for r in res]); m = [np.mean([r[i] for r in res], 0) for i in (1, 2, 3)]
        print(f"shots={shots:2d} rot={'Y' if rot else 'N'} cap={min(cap, res[0][-1]):6d} proj={'128' if proj else '384'}  AUROC={a:.3f}  "
              + "  ".join(f"x{f}: catch {c:.0%}/false-rej {fr:.0%}" for f, (c, fr) in zip((0.8, 0.9, 1.0), m)), flush=True)

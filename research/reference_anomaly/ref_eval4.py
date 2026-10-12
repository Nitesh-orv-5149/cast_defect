"""Round 4: auto threshold = leave-one-out score of each good reference vs the other references' views.
1 shot: score 22.5-degree-offset views (not in the bank) of the single reference."""
import random, sys
from pathlib import Path
import numpy as np, torch
import torchvision.transforms.functional as TF
sys.path.insert(0, str(Path(__file__).parent))
from ref_eval import ROOT, OUT, load, patches, resnet18, auroc

torch.set_grad_enabled(False)
def views(x, off=0.0):
    out = [TF.rotate(x, 45 * k + off) for k in range(8)]; return torch.cat(out + [o.flip(3) for o in out])
ok_train = sorted((ROOT / "train/ok_front").glob("*.jpeg"))
rng = random.Random(0)
test_ok = rng.sample(sorted((ROOT / "test/ok_front").glob("*.jpeg")), 200)
test_def = rng.sample(sorted((ROOT / "test/def_front").glob("*.jpeg")), 200)
run = resnet18()
tq = {k: torch.cat([patches(run, load(ps[i:i+40])) for i in range(0, len(ps), 40)]) for k, ps in [("ok", test_ok), ("def", test_def)]}
C = tq["ok"].shape[-1]
def mk_bank(per):  # per: list of (n_views*784, C)
    b = torch.cat(per); return b[torch.randperm(b.shape[0])[:60000]] if b.shape[0] > 60000 else b
lines = []
for shots in (1, 3, 5, 10, 20):
    for seed in (1, 2):
        random.seed(seed); torch.manual_seed(seed)
        refs = load(random.sample(ok_train, shots))
        per = [patches(run, views(refs[i:i+1])).reshape(-1, C) for i in range(shots)]
        bank = mk_bank(per)
        sc = lambda q, b: torch.cdist(q, b).min(1).values.topk(10).values.mean().item()
        if shots == 1:
            cal = [sc(q, bank) for q in patches(run, views(refs[:1], 22.5))]
        else:
            P = patches(run, refs)
            cal = [sc(P[i], mk_bank(per[:i] + per[i+1:])) for i in range(shots)]
        so = np.array([sc(q, bank) for q in tq["ok"]]); sd = np.array([sc(q, bank) for q in tq["def"]])
        res = [f"x{m}: catch {(sd > max(cal)*m).mean():.0%} / false-rej {(so > max(cal)*m).mean():.0%}" for m in (1.0, 1.1, 1.2)]
        l = f"shots={shots:2d} seed={seed} AUROC={auroc(so, sd):.3f} | " + " | ".join(res)
        print(l, flush=True); lines.append(l)
(OUT / "results4.txt").write_text("\n".join(lines))

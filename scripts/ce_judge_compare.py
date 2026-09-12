"""Judge-vs-judge comparison on shared items.

  python3 ce_judge_compare.py judge_longrun_raw.jsonl judge_gemini_raw.jsonl

Item-level: for every item rated by both judges (primary role), agreement
(Spearman rho, mean |diff|, offset), overall / by arm / by domain.
Arm-level: mean delta of judge B's rating minus judge A's on the same item,
by arm -- if the two judges agree on the *effects*, these are all the same
constant (the scale offset); an arm whose offset differs is one the judges
see differently. Also judge B's own arm means vs judge A's, for the ordering.
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

A, B = sys.argv[1], sys.argv[2]


def load(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("role") == "primary" and "rating" in r:
            out[r["uid"]] = r
    return out


a, b = load(A), load(B)
shared = sorted(set(a) & set(b))
print(f"{len(a)} items in A, {len(b)} in B, {len(shared)} shared")
ra = np.array([a[u]["rating"] for u in shared]); rb = np.array([b[u]["rating"] for u in shared])
print(f"\nOVERALL: Spearman rho {stats.spearmanr(ra, rb).correlation:.2f}, "
      f"mean |diff| {np.mean(np.abs(ra-rb)):.2f}, exact {np.mean(ra==rb):.0%}, "
      f"within 1 {np.mean(np.abs(ra-rb)<=1):.0%}, means A {ra.mean():.2f} B {rb.mean():.2f} "
      f"(offset B-A {rb.mean()-ra.mean():+.2f})")
arms = ["reference", "resample", "lam1.0", "lam0.2", "raw4bit", "raw3bit", "d15_B16", "d15_noref"]
print(f"\n{'arm':>10} {'n':>4} {'mean A':>7} {'mean B':>7} {'B-A':>6} {'rho':>5} {'|diff|':>7}")
for arm in arms:
    idx = [i for i, u in enumerate(shared) if a[u]["arm"] == arm]
    if len(idx) < 3:
        continue
    x, y = ra[idx], rb[idx]
    rho = stats.spearmanr(x, y).correlation if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    print(f"{arm:>10} {len(idx):4d} {x.mean():7.2f} {y.mean():7.2f} {y.mean()-x.mean():+6.2f} {rho:5.2f} {np.mean(np.abs(x-y)):7.2f}")
print(f"\n{'domain':>10} {'n':>4} {'mean A':>7} {'mean B':>7} {'B-A':>6} {'rho':>5}")
for d in ["prose", "code", "arith", "zh", "hard"]:
    idx = [i for i, u in enumerate(shared) if a[u]["domain"] == d]
    if len(idx) < 3:
        continue
    x, y = ra[idx], rb[idx]
    rho = stats.spearmanr(x, y).correlation if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    print(f"{d:>10} {len(idx):4d} {x.mean():7.2f} {y.mean():7.2f} {y.mean()-x.mean():+6.2f} {rho:5.2f}")
# B's arm effects relative to the reference arm, using B's own reference ratings where available
refB = {(r["domain"], r["prompt_idx"]): r["rating"] for r in b.values() if r["arm"] == "reference"}
refA = {(r["domain"], r["prompt_idx"]): r["rating"] for r in a.values() if r["arm"] == "reference"}
print(f"\nARM EFFECTS (rating minus the reference rating on the same prompt), judge B vs judge A on B's items:")
print(f"{'arm':>10} {'n':>4} {'delta B':>8} {'delta A':>8}")
for arm in arms[1:]:
    dB, dA = [], []
    for u, r in b.items():
        k = (r["domain"], r["prompt_idx"])
        if r["arm"] == arm and k in refA:
            dA.append(a[u]["rating"] - refA[k]) if u in a else None
            if k in refB:
                dB.append(r["rating"] - refB[k])
    if dA:
        print(f"{arm:>10} {len(dA):4d} {np.mean(dB) if dB else float('nan'):+8.2f} {np.mean(dA):+8.2f}   (B has own reference on {len(dB)} of these)")
# disagreements > 2 points
big = [(u, a[u]["rating"], b[u]["rating"], b[u].get("reason", "")[:80]) for u in shared if abs(a[u]["rating"] - b[u]["rating"]) >= 3]
print(f"\nitems differing by >= 3 points: {len(big)}")
for t in big[:12]:
    print("  ", t)

"""Close Enough -- analysis of the judged run.

Inputs:  judge_longrun_raw.jsonl   (from judge_longrun.py, ITEMS=ce_items*.json)
         ce_items_checked.json     (from ce_check.py; falls back to ce_items.json)
Output:  printed report + ce_results.json

Design (fixed before looking at the data):
  * Baseline per prompt = mean of the primary ratings of `reference` and
    `resample`. Every measured arm is compared to that baseline.
  * Exchangeability test = resample minus reference (mean must be 0).
  * Per arm: mean delta, s.d., 95% CI, and a TOST equivalence verdict at the
    pre-stated bound EQ_BOUND (default 0.3 rating points): equivalent iff the
    90% CI lies inside [-EQ_BOUND, +EQ_BOUND]. Pooled and per domain.
  * Variance decomposition: judge (duplicate) variance, per-rating residual,
    prompt effect (ICC of deltas across arms).
  * Joins: judged delta vs EAR (speculative arms); judged rating vs external
    correctness / agreement with reference (arith, hard).
  * Prediction scorecard against the registered values in PREDICTIONS.
"""
import json, math, os
from collections import defaultdict
import numpy as np

RAW = os.environ.get("RAW", "judge_longrun_raw.jsonl")
ITEMS = os.environ.get("ITEMS", "ce_items_checked.json"
                       if os.path.exists("ce_items_checked.json")
                       else "ce_items.json")
EQ_BOUND = float(os.environ.get("EQ_BOUND", "0.3"))
BASE_ARMS = ("reference", "resample")

# registered before the run (pooled delta vs baseline unless noted)
PREDICTIONS = {
    "resample": (0.00, "null by exchangeability (vs reference)"),
    "lam1.0": (0.00, "null by proof; |delta| < 0.05"),
    "lam0.2": (-0.05, "[-0.2, +0.1]"),
    "raw4bit": (-0.10, "pooled; zh -0.45; arith 0.00"),
    "raw3bit": (-0.80, "[-1.5, -0.4]"),
    "d15_B16": (-0.55, "greedy anchor -0.59"),
    "d15_noref": (-1.00, "greedy anchor -1.05"),
}

try:
    from scipy import stats

    def tq(p, df):
        return float(stats.t.ppf(p, df))

    def spearman(x, y):
        r = stats.spearmanr(x, y)
        return float(r.correlation), float(r.pvalue)
except ImportError:
    def tq(p, df):
        return 1.96 if p > 0.97 else 1.645

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def ci(d, level=0.95):
    d = np.asarray(d, float)
    n = len(d)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    t = tq(1 - (1 - level) / 2, n - 1)
    return m, m - t * se, m + t * se


def fmt_row(name, d):
    m, lo, hi = ci(d)
    _, lo90, hi90 = ci(d, 0.90)
    sig = "YES" if (hi < 0 or lo > 0) else "no"
    eq = ("EQUIV" if (lo90 > -EQ_BOUND and hi90 < EQ_BOUND) else
          "not shown" if np.isfinite(lo90) else "--")
    return (f"{name:>12} {len(d):4d} {m:+7.2f} {np.std(d, ddof=1):6.2f} "
            f"[{lo:+6.2f},{hi:+6.2f}] {sig:>4} {eq:>10}")


def main():
    recs = [json.loads(l) for l in open(RAW)]
    recs = [r for r in recs if "rating" in r]
    items = {(r["domain"], r["prompt_idx"], r["arm"]): r
             for r in json.load(open(ITEMS))}
    prim = {r["uid"]: r for r in recs if r["role"] == "primary"}
    dup = {r["uid"]: r for r in recs if r["role"] == "dup"}
    sec = {r["uid"]: r for r in recs if r["role"] == "second"}
    print(f"{len(recs)} judgements: {len(prim)} primary, {len(dup)} duplicate, "
          f"{len(sec)} second-opinion; items file {ITEMS}")

    # ---------------------------------------------------------- reliability
    print("\n" + "=" * 88 + "\nJUDGE RELIABILITY\n" + "=" * 88)
    dd = [prim[u]["rating"] - dup[u]["rating"] for u in dup if u in prim]
    if dd:
        print(f"  same item twice (n={len(dd)}): mean |diff| "
              f"{np.mean(np.abs(dd)):.3f}, exact {np.mean(np.array(dd)==0):.0%}, "
              f"s.d. of difference {np.std(dd, ddof=1):.3f} "
              f"-> judge-only s.d. per rating {np.std(dd, ddof=1)/math.sqrt(2):.3f}")
    d2 = [(prim[u]["rating"], sec[u]["rating"]) for u in sec if u in prim]
    if d2:
        a, b = zip(*d2)
        print(f"  primary vs second model (n={len(d2)}): mean |diff| "
              f"{np.mean(np.abs(np.array(a)-np.array(b))):.2f}, means "
              f"{np.mean(a):.2f} / {np.mean(b):.2f}")

    # --------------------------------------------------------------- tables
    rating = defaultdict(dict)            # (dom, idx) -> arm -> rating
    for r in prim.values():
        rating[(r["domain"], r["prompt_idx"])][r["arm"]] = r["rating"]
    keys = sorted(k for k, v in rating.items()
                  if all(a in v for a in BASE_ARMS))
    base = {k: np.mean([rating[k][a] for a in BASE_ARMS]) for k in keys}
    arms = sorted({a for v in rating.values() for a in v} - set(BASE_ARMS))
    doms = sorted({k[0] for k in keys})
    print(f"\n{len(keys)} prompts with both baseline arms; domains {doms}")

    refs = [rating[k]["reference"] for k in keys]
    print(f"\nreference mean {np.mean(refs):.2f}; ceiling (rated 7) "
          f"{np.mean(np.array(refs)==7):.0%}; by domain: "
          + ", ".join(f"{d} {np.mean([rating[k]['reference']==7 for k in keys if k[0]==d]):.0%}"
                      for d in doms))

    print("\n" + "=" * 88 + "\nEXCHANGEABILITY TEST: resample minus reference "
          "(true mean is 0 by symmetry)\n" + "=" * 88)
    hdr = (f"{'arm':>12} {'n':>4} {'delta':>7} {'sd':>6} {'95% CI':>15} "
           f"{'sig':>4} {'TOST±'+str(EQ_BOUND):>10}")
    print(hdr)
    ex = [rating[k]["resample"] - rating[k]["reference"] for k in keys]
    print(fmt_row("resample", ex))
    for d in doms:
        print(fmt_row(f"  {d}", [rating[k]["resample"] - rating[k]["reference"]
                                 for k in keys if k[0] == d]))

    print("\n" + "=" * 88 + "\nMEASURED ARMS vs DUAL-REFERENCE BASELINE\n"
          + "=" * 88)
    print(hdr)
    deltas = {}
    for a in arms:
        deltas[a] = {k: rating[k][a] - base[k] for k in keys if a in rating[k]}
        print(fmt_row(a, list(deltas[a].values())))
    print("\nBY DOMAIN (delta, 95% CI):")
    print(f"{'arm':>12}" + "".join(f"{d:>22}" for d in doms))
    for a in arms:
        row = f"{a:>12}"
        for d in doms:
            v = [x for k, x in deltas[a].items() if k[0] == d]
            m, lo, hi = ci(v)
            row += f"  {m:+5.2f} [{lo:+5.2f},{hi:+5.2f}]" if len(v) >= 3 else f"{'--':>22}"
        print(row)

    # ---------------------------------------------------- variance structure
    print("\n" + "=" * 88 + "\nVARIANCE STRUCTURE\n" + "=" * 88)
    common = [k for k in keys if all(k in deltas[a] for a in arms)]
    M = np.array([[deltas[a][k] for a in arms] for k in common])
    pm = M.mean(1)
    ms_p = M.shape[1] * ((pm - M.mean()) ** 2).sum() / (M.shape[0] - 1)
    ms_w = ((M - pm[:, None]) ** 2).sum() / (M.shape[0] * (M.shape[1] - 1))
    var_p = max(0.0, (ms_p - ms_w) / M.shape[1])
    jv = np.var(dd, ddof=1) / 2 if dd else float("nan")
    print(f"  judge-only variance per rating      {jv:.3f}")
    print(f"  residual variance per paired delta  {ms_w:.3f}")
    print(f"  prompt-effect variance (shared)     {var_p:.3f}  ICC {var_p/(var_p+ms_w):.2f}")
    print(f"  paired-delta s.d. by domain: "
          + ", ".join(f"{d} {np.std([x for a in arms for k, x in deltas[a].items() if k[0]==d], ddof=1):.2f}"
                      for d in doms))

    # ------------------------------------------------------------- joins
    print("\n" + "=" * 88 + "\nJUDGED DELTA vs MECHANICAL EAR (speculative arms)\n"
          + "=" * 88)
    for a in arms:
        xs, ys = [], []
        for k, dl in deltas[a].items():
            it = items.get((k[0], k[1], a))
            if it and "ear_mean" in it:
                xs.append(it["ear_mean"]); ys.append(dl)
        if len(xs) > 10:
            rho, p = spearman(xs, ys)
            print(f"  {a:>8}: n={len(xs)} mean EAR {np.mean(xs):.3f}, "
                  f"Spearman(EAR, delta) = {rho:+.2f} (p={p:.2g})")

    print("\n" + "=" * 88 + "\nJUDGED RATING vs CORRECTNESS (arith, hard)\n"
          + "=" * 88)
    for col in ("correct", "agrees_ref"):
        rows = []
        for (d, i, a), it in items.items():
            if it.get(col) is None or (d, i) not in rating or a not in rating[(d, i)]:
                continue
            rows.append((it[col], rating[(d, i)][a], d))
        if rows:
            for d in ("arith", "hard"):
                t = [r for c, r, dd_ in rows if c and dd_ == d]
                f = [r for c, r, dd_ in rows if not c and dd_ == d]
                if t or f:
                    print(f"  {col:>10} {d:>6}: rating when True "
                          f"{np.mean(t) if t else float('nan'):.2f} (n={len(t)}), "
                          f"when False {np.mean(f) if f else float('nan'):.2f} (n={len(f)})")
    # items the judge rated >= 6 but the checker says wrong
    miss = [(d, i, a, rating[(d, i)][a]) for (d, i, a), it in items.items()
            if it.get("correct") is False and (d, i) in rating
            and a in rating[(d, i)] and rating[(d, i)][a] >= 6]
    print(f"  wrong-by-checker but judged >= 6: {len(miss)}"
          + (f"  e.g. {miss[:5]}" if miss else ""))

    # --------------------------------------------------------- scorecard
    print("\n" + "=" * 88 + "\nPREDICTION SCORECARD (registered before the run)\n"
          + "=" * 88)
    print(f"{'arm':>12} {'predicted':>10} {'measured':>9} {'95% CI':>16} {'verdict':>8}  note")
    out = {}
    for a, (pred, note) in PREDICTIONS.items():
        v = ex if a == "resample" else list(deltas.get(a, {}).values())
        if len(v) < 3:
            continue
        m, lo, hi = ci(v)
        verdict = "inside" if lo <= pred <= hi else "OUTSIDE"
        print(f"{a:>12} {pred:+10.2f} {m:+9.2f} [{lo:+6.2f},{hi:+6.2f}] {verdict:>8}  {note}")
        out[a] = dict(pred=pred, mean=m, lo=lo, hi=hi, n=len(v))
    json.dump(dict(eq_bound=EQ_BOUND, n_prompts=len(keys), arms=out,
                   judge_sd=float(np.std(dd, ddof=1) / math.sqrt(2)) if dd else None,
                   icc=var_p / (var_p + ms_w) if common else None),
              open("ce_results.json", "w"), indent=1)
    print("\nwrote ce_results.json")


if __name__ == "__main__":
    main()

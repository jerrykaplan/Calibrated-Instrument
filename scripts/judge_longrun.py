"""Blind judging of the cache-repair runs.

Every quality claim so far rests on two crude automatic measures -- failure to
terminate and repeated 8-grams -- plus a handful of examples read by hand. The
perplexity analysis showed that responses which failed to terminate scored
BETTER on perplexity than intact ones (1.208 against 1.595), so automatic
measures cannot be trusted on this failure mode. This is the check.

Design:
  * Blind. The judge sees only the prompt and the response. No condition, no
    arm, no density, no depth.
  * Shuffled. Items from every condition and both depths are interleaved in one
    randomised order, so judge drift over the batch cannot align with condition.
  * References included. The unmodified model's own output is judged on the
    same scale, giving a baseline rather than an absolute standard.
  * Duplicated. A fraction of items are judged twice at different positions in
    the shuffle. Disagreement between the two scores of the SAME item measures
    the instrument, and bounds how much of any observed difference is real.
  * Second opinion. A subset is also judged by a stronger model. If the two
    agree at the level the weaker one agrees with itself, the weaker one is
    adequate.

Resumable: results append to judge_raw.jsonl and completed ids are skipped.

  export ANTHROPIC_API_KEY=...      (or: read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY)
  python3 judge.py                  # everything
  python3 judge.py 60               # 60 items, for a costed trial
"""
import json, os, random, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ITEMS = os.environ.get("ITEMS", "longrun_items.json")
JUDGE = os.environ.get("JUDGE_MODEL", "claude-sonnet-5")
JUDGE2 = os.environ.get("JUDGE2_MODEL", "claude-opus-5")
DUP_FRAC = float(os.environ.get("DUP_FRAC", "0.20"))
SECOND_FRAC = float(os.environ.get("SECOND_FRAC", "0.10"))
WORKERS = int(os.environ.get("WORKERS", "6"))
RAW = "judge_longrun_raw.jsonl"
SEED = 20260825
URL = "https://api.anthropic.com/v1/messages"
KEY = "".join(os.environ.get("ANTHROPIC_API_KEY", "").split())

RUBRIC = """You are rating the quality of a response produced by a language model.

Rate how well the response answers the request, on this scale:

7  Excellent. Complete, correct, well-formed. Nothing a reader would object to.
6  Good. Minor awkwardness or a small omission, but fully usable.
5  Acceptable. Noticeable rough edges: slight redundancy, a clumsy sentence, a
   small factual slip, or an ending that trails rather than concludes.
4  Mediocre. Real problems: repeats itself, drifts off the question, contains a
   clear factual or logical error, or is visibly incomplete.
3  Poor. Substantially fails the request, though still recognisable as an
   attempt at it.
2  Bad. Largely incoherent, or degenerates part-way through.
1  Broken. Degenerate repetition, no meaningful content, or unusable output.

Also decide, separately: did the response DERAIL? Derailed means it lost the
thread -- looping on a word or phrase, abandoning the topic, producing garbled
or unrelated text, or running on without ever concluding. A response can be
mediocre without derailing.

And decide: did the response RECOVER? If the response contains a damaged,
garbled, or confused stretch but returns to coherent, on-task output afterward
and concludes properly, recovered is true. If it never goes wrong at all,
recovered is true. If it goes wrong and stays wrong, recovered is false.

If the response contains code, judge whether the code is syntactically valid
and does what was asked. Unbalanced brackets, undefined names, or truncated
blocks are serious faults.

Respond with JSON only, no other text, no markdown fences:
{"rating": <1-7>, "derailed": <true|false>, "recovered": <true|false>, "reason": "<at most 15 words>"}"""


_shown = [False]


def call(model, prompt_text, response_text, tries=5):
    body = json.dumps({
        "model": model, "max_tokens": 8000, "system": RUBRIC,
        "messages": [{"role": "user", "content":
                      f"<request>\n{prompt_text}\n</request>\n\n"
                      f"<response>\n{response_text}\n</response>"}],
    }).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(
                URL, data=body,
                headers={"x-api-key": KEY, "anthropic-version": "2023-06-01",
                         "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            txt = "".join(c.get("text", "") for c in out["content"])
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip()).strip()
            try:
                j = json.loads(txt)
            except json.JSONDecodeError as pe:
                if not txt.strip():
                    raise RuntimeError(
                        f"empty judge text; stop_reason={out.get('stop_reason')} "
                        f"blocks={[c.get('type') for c in out['content']]} "
                        f"usage={out.get('usage')}") from pe
                mo = re.search(r"\{.*\}", txt, re.S)     # any embedded object
                if mo:
                    j = json.loads(mo.group(0))
                else:                                      # rebuild from fields
                    r_ = re.search(r'"rating"\s*:\s*(\d)', txt)
                    d_ = re.search(r'"derailed"\s*:\s*(true|false)', txt)
                    if not (r_ and d_):
                        raise
                    rs = re.search(r'"reason"\s*:\s*"([^"]*)', txt)
                    j = {"rating": int(r_.group(1)),
                         "derailed": d_.group(1) == "true",
                         "reason": (rs.group(1) if rs else "") + " [truncated]"}
            return {"rating": int(j["rating"]), "derailed": bool(j["derailed"]),
                    "recovered": bool(j.get("recovered", True)),
                    "reason": str(j.get("reason", ""))[:200]}
        except urllib.error.HTTPError as e:
            body = e.read()[:400].decode(errors="replace")
            if not _shown[0]:
                _shown[0] = True
                print(f"\n  first API error: HTTP {e.code}\n  {body}\n",
                      flush=True)
            if e.code in (429, 500, 502, 503, 529) and t < tries - 1:
                time.sleep(2 ** t + random.random() * 2); continue
            return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            if t < tries - 1:
                time.sleep(2 ** t + random.random()); continue
            return {"error": f"{type(e).__name__}: {e}"}


def load_items():
    items = []
    for x in json.load(open(ITEMS)):
        if "text" not in x:
            continue
        items.append(dict(
            uid=f"{x['domain']}_{x['prompt_idx']}_{x['arm']}",
            domain=x["domain"], prompt_idx=x["prompt_idx"],
            arm=x["arm"], method=x.get("method"), depth=x.get("depth"),
            prompt=x["prompt"], text=x["text"],
            hit_cap=x.get("hit_cap", 0),
            len_ratio=x.get("len_ratio"), div_frac=x.get("div_frac")))
    return items


def main():
    if not KEY:
        raise SystemExit("ANTHROPIC_API_KEY is not set. "
                         "run:  read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY")
    print(f"key: {len(KEY)} chars, starts {KEY[:12]!r}", flush=True)
    pre = call(JUDGE, "Say ok.", "ok")
    if "error" in pre:
        raise SystemExit(f"preflight failed against {JUDGE}: {pre['error']}")
    print(f"preflight OK ({JUDGE} returned rating "
          f"{pre.get('rating')})", flush=True)

    rng = random.Random(SEED)
    items = load_items()
    if LIMIT:
        rng.shuffle(items); items = items[:LIMIT]
    print(f"{len(items)} items to judge", flush=True)

    jobs = [dict(it, job=f"{it['uid']}#a", model=JUDGE, role="primary")
            for it in items]
    for it in rng.sample(items, int(len(items) * DUP_FRAC)):
        jobs.append(dict(it, job=f"{it['uid']}#b", model=JUDGE, role="dup"))
    for it in rng.sample(items, int(len(items) * SECOND_FRAC)):
        jobs.append(dict(it, job=f"{it['uid']}#2", model=JUDGE2, role="second"))
    rng.shuffle(jobs)

    done, failed = set(), 0
    if os.path.exists(RAW):
        keep = []
        for line in open(RAW):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "rating" in r:           # only successes count as done
                done.add(r["job"]); keep.append(line)
            else:
                failed += 1
        if failed:
            # drop failed records so they are retried and not counted later
            with open(RAW, "w") as f:
                f.writelines(keep)
        print(f"resuming: {len(done)} already judged"
              + (f", {failed} earlier failures dropped and will be retried"
                 if failed else ""), flush=True)
    todo = [j for j in jobs if j["job"] not in done]
    print(f"{len(todo)} calls to make "
          f"({sum(1 for j in todo if j['role']=='primary')} primary, "
          f"{sum(1 for j in todo if j['role']=='dup')} duplicate, "
          f"{sum(1 for j in todo if j['role']=='second')} second-opinion)",
          flush=True)

    t0 = [time.time()], [0]
    out = open(RAW, "a")

    def work(j):
        r = call(j["model"], j["prompt"], j["text"])
        rec = {k: j[k] for k in ("job", "uid", "domain", "prompt_idx",
                                 "arm", "method", "depth", "model", "hit_cap",
                                 "len_ratio", "div_frac")}
        rec["role"] = j["role"]
        rec.update(r)
        return rec

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rec in ex.map(work, todo):
            out.write(json.dumps(rec) + "\n"); out.flush()
            t0[1][0] += 1
            if t0[1][0] % 25 == 0:
                el = time.time() - t0[0][0]
                print(f"  {t0[1][0]}/{len(todo)}  {el:.0f}s  "
                      f"eta {(len(todo)-t0[1][0])*el/t0[1][0]/60:.0f} min", flush=True)
    out.close()
    analyse(items)


def analyse(items):
    import statistics as st, math
    recs = [json.loads(l) for l in open(RAW)]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "rating" in r]
    print(f"\n{len(recs)} judgements, {len(errs)} errors")
    prim = {r["uid"]: r for r in recs if r["job"].endswith("#a")}
    dup = {r["uid"]: r for r in recs if r["job"].endswith("#b")}
    sec = {r["uid"]: r for r in recs if r["job"].endswith("#2")}

    print("\n" + "=" * 92)
    print("JUDGE RELIABILITY")
    print("=" * 92)
    both = [(prim[u]["rating"], dup[u]["rating"]) for u in dup if u in prim]
    if both:
        d = [abs(a - b) for a, b in both]
        print(f"  same item twice (n={len(both)}): mean |diff| {st.mean(d):.2f}, "
              f"exact {sum(x==0 for x in d)/len(d):.0%}, within 1 "
              f"{sum(x<=1 for x in d)/len(d):.0%}")
    b2 = [(prim[u]["rating"], sec[u]["rating"]) for u in sec if u in prim]
    if b2:
        d2 = [abs(a - b) for a, b in b2]
        print(f"  {JUDGE} vs {JUDGE2} (n={len(b2)}): mean |diff| {st.mean(d2):.2f}, "
              f"means {st.mean([a for a,_ in b2]):.2f} / "
              f"{st.mean([b for _,b in b2]):.2f}")

    base = {(r["domain"], r["prompt_idx"]): r["rating"]
            for r in prim.values() if r["arm"] == "reference"}
    print("\n" + "=" * 92)
    print("RATINGS  (paired against the SAME PROMPT's unmodified reference)")
    print("  Cache reconstructed from the FIRST generated token, every position,")
    print("  no token corruption anywhere. Responses 400-600 tokens.")
    print("=" * 92)
    print(f"{'arm':>12s} {'n':>4s} {'rating':>7s} {'derail':>7s} {'recov':>6s} "
          f"{'delta':>7s} {'95% CI':>17s} {'sig':>4s}")
    ref = [r["rating"] for r in prim.values() if r["arm"] == "reference"]
    if ref:
        print(f"{'reference':>12s} {len(ref):4d} {st.mean(ref):7.2f} "
              f"{st.mean([r['derailed'] for r in prim.values() if r['arm']=='reference']):7.2f} "
              f"{st.mean([r.get('recovered',True) for r in prim.values() if r['arm']=='reference']):6.2f}")
    arms = sorted({r["arm"] for r in prim.values() if r["arm"] != "reference"},
                  key=lambda a: (a.split("_")[0], -int(re.search(r"_d(\d+)", a).group(1)) if re.search(r"_d(\d+)", a) else 0, a))
    for a in arms:
        g = [r for r in prim.values() if r["arm"] == a]
        v = [r["rating"] for r in g]
        d = [r["rating"] - base[(r["domain"], r["prompt_idx"])] for r in g
             if (r["domain"], r["prompt_idx"]) in base]
        if len(d) < 3:
            continue
        mm = st.mean(d)
        se = st.stdev(d) / math.sqrt(len(d))
        lo, hi = mm - 1.96 * se, mm + 1.96 * se
        print(f"{a:>12s} {len(v):4d} {st.mean(v):7.2f} "
              f"{st.mean([r['derailed'] for r in g]):7.2f} "
              f"{st.mean([r.get('recovered',True) for r in g]):6.2f} "
              f"{mm:+7.2f} [{lo:+7.2f},{hi:+7.2f}] "
              f"{'YES' if (hi<0 or lo>0) else 'no':>4s}")

    print("\n" + "=" * 92)
    print("BY DOMAIN (delta vs reference)")
    print("=" * 92)
    doms = sorted({r["domain"] for r in prim.values()})
    print(f"{'arm':>12s}" + "".join(f"{x:>11s}" for x in doms))
    for a in arms:
        row = f"{a:>12s}"
        for dom in doms:
            d = [r["rating"] - base[(r["domain"], r["prompt_idx"])]
                 for r in prim.values() if r["arm"] == a and r["domain"] == dom
                 and (r["domain"], r["prompt_idx"]) in base]
            row += f"{st.mean(d):+11.2f}" if len(d) >= 2 else f"{'--':>11s}"
        print(row)

    print("\n" + "=" * 92)
    print("AUTOMATIC MEASURES vs THE JUDGE")
    print("=" * 92)
    cap = [r["rating"] for r in prim.values() if r["hit_cap"] == 1]
    noc = [r["rating"] for r in prim.values()
           if r["hit_cap"] == 0 and r["arm"] != "reference"]
    if cap:
        print(f"  never terminated (n={len(cap)}): mean {st.mean(cap):.2f}")
    if noc:
        print(f"  terminated       (n={len(noc)}): mean {st.mean(noc):.2f}")
    miss = [r for r in prim.values() if r["hit_cap"] == 0 and r["rating"] <= 3
            and r["arm"] != "reference"]
    print(f"  terminated but judged <=3: {len(miss)}")

    byuid = {it["uid"]: it for it in items}
    lines = []

    def show(title, rows, n=3):
        lines.append("\n" + "=" * 92)
        lines.append(title)
        lines.append("=" * 92)
        seen = set()
        for r in rows:
            if r["uid"] in seen or len(seen) >= n:
                continue
            seen.add(r["uid"])
            it = byuid.get(r["uid"])
            if not it:
                continue
            lines.append(f"\n[{r['uid']}] RATING {r['rating']} "
                         f"derailed={r['derailed']} recovered={r.get('recovered')}")
            lines.append(f"  judge: {r['reason']}")
            lines.append(f"  PROMPT: {it['prompt'][:100]}")
            t = it["text"]
            lines.append(f"  first 240: {t[:240]}")
            if len(t) > 800:
                lines.append(f"  middle 240: {t[len(t)//2:len(t)//2+240]}")
            lines.append(f"  last 200: ...{t[-200:]}")

    for a in ["reference"] + arms:
        g = sorted([r for r in prim.values() if r["arm"] == a],
                   key=lambda r: r["rating"])
        if len(g) >= 3:
            show(f"{a} -- worst / median / best of {len(g)}",
                 [g[0], g[len(g)//2], g[-1]], 3)
    if miss:
        show("TERMINATED BUT JUDGED <=3", sorted(miss, key=lambda r: r["rating"]), 5)

    open("judge_longrun_samples.txt", "w").write("\n".join(lines))
    print("\n".join(lines[:80]))
    print("\n... full samples in judge_longrun_samples.txt")
    json.dump(list(prim.values()), open("judge_longrun_summary.json", "w"), indent=1)
    print("wrote judge_longrun_raw.jsonl, judge_longrun_summary.json, "
          "judge_longrun_samples.txt")


if __name__ == "__main__":
    main()

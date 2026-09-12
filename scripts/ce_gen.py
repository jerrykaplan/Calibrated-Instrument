"""Close Enough -- powered generation run (all arms, all domains, one machine).

Per prompt, nine generations from Qwen2.5-7B-Instruct at tau=0.7 (pure
temperature; no top-p / top-k), in this fixed order:

  reference    bf16 plain sample, seed A
  resample     bf16 plain sample, seed B      (null by exchangeability;
                                               also half of the baseline)
  lam1.0       NF4 draft + bf16 verify, strict rejection rule (null by proof)
  lam0.2       same loop, accept with prob min(1, r/0.2)   (stressed end)
  raw4bit      NF4 alone, no verification
  raw3bit      HQQ 3-bit alone, no verification            (stressed end)
  d15_B16      early exit at layer 15 with fitted repair map, exact refill
               every 16 tokens (near-threshold anchor, -0.59 greedy)
  d15_noref    early exit at layer 15 with repair map, no refill
               (detectable anchor, -1.05 greedy)

Domains and counts (env NS): prose 40, code 60, arith 40, zh 40, hard 40.
Hard prompts come from prompts_hard.json (math items carry a numeric truth
and the 'Final answer:' instruction; code items carry assert tests that a
separate checker runs).

Per record: text, token count, wall seconds, rep8, hit_cap, final answer
parse, prompt-integer integrity flag (arith/hard-math), and for the
speculative arms the per-window acceptance statistics plus mean EAR
(expected acceptance rate = sum_k min(p_t(k), p_d(k)), i.e. 1 - TV).

  # pod (RTX 5090), after: pip install transformers accelerate bitsandbytes hqq
  SELFTEST=1 HF_HOME=/workspace/hf FITS=/workspace/nway_7b python3 ce_gen.py
  HF_HOME=/workspace/hf FITS=/workspace/nway_7b python3 ce_gen.py

Resumable: one JSON per prompt in OUT/; finished prompts are skipped.
Output for judging: ce_items.json (judge_longrun.py schema).
"""
import json, os, re, time, urllib.request
import numpy as np
import torch

KEY = os.environ.get("KEY", "q7")
MODELS = {"q15": "Qwen/Qwen2.5-1.5B-Instruct", "q7": "Qwen/Qwen2.5-7B-Instruct",
          "l8": "meta-llama/Llama-3.1-8B-Instruct"}
EOS = set()          # filled at load: all end-of-sequence ids (Llama has several)
OUT = os.environ.get("OUT", "ce_out")
ITEMS_OUT = os.environ.get("ITEMS_OUT", "ce_items.json")
FITS = os.environ.get("FITS", "/workspace/nway_7b")
FITPAT = os.environ.get("FITPAT", "fit_d{D}.npz")
DEPTH = int(os.environ.get("DEPTH", "15"))
REFILL_B = int(os.environ.get("REFILL_B", "16"))
B = int(os.environ.get("B", "8"))                 # speculative draft window
TAU = float(os.environ.get("TAU", "0.7"))
MAXNEW = int(os.environ.get("MAXNEW", "700"))
SEED = int(os.environ.get("SEED", "0"))
LAMS = tuple(float(x) for x in os.environ.get("LAMS", "1.0,0.2").split(","))
NS = dict((k, int(v)) for k, v in
          (x.split(":") for x in os.environ.get(
              "NS", "prose:40,code:60,arith:40,zh:40,hard:40").split(",")))
ARMS = os.environ.get("ARMS", "reference,resample,lam,raw4bit,raw3bit,"
                      "refill,noref").split(",")
SELFTEST = os.environ.get("SELFTEST", "") == "1"
HQQ_BITS = int(os.environ.get("HQQ_BITS", "3"))
HQQ_GROUP = int(os.environ.get("HQQ_GROUP", "64"))
STD = {"prose": "prompts.json", "code": "prompts_code.json",
       "arith": "prompts_arith.json", "zh": "prompts_zh.json"}
HARD = "prompts_hard.json"
RAW = ("https://raw.githubusercontent.com/jerrykaplan/"
       "question-conditioned-early-exit/main/prompts/")
DOM_IDX = {"prose": 0, "code": 1, "arith": 2, "zh": 3, "hard": 4}


# ------------------------------------------------------------------ utilities
def fetch():
    for fn in STD.values():
        if not os.path.exists(fn):
            urllib.request.urlretrieve(RAW + fn, fn)


def gb():
    return torch.cuda.memory_allocated() / 1e9


def rep8(t):
    w = t.split()
    if len(w) < 20:
        return 0.0
    g = [" ".join(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1 - len(set(g)) / len(g)


def parse_final(text):
    """'Final answer: <value>' if present (last occurrence), else the last
    number in the text. Returns a string (may be a fraction like 73/648)."""
    text = re.sub(r"\\d?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
    m = re.findall(r"[Ff]inal [Aa]nswer\s*\**\s*[:：][^\d\n-]*(-?\d[\d,./ ]*)",
                   text)
    if m:
        return m[-1].replace(",", "").replace(" ", "").rstrip("./")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def prompt_ints(t):
    return set(re.findall(r"\d+", t))


def rmsnorm(x, w, eps=1e-6):
    return x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + eps) * w


# ------------------------------------------------------------ cache handling
def cache_kv(cache):
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return [(k, v) for k, v in cache]


def crop_cache(cache, n_keep):
    n_now = cache.get_seq_length()
    if n_now <= n_keep:
        return cache
    if hasattr(cache, "crop"):
        cache.crop(n_keep - n_now)        # negative form works on all versions
        return cache
    for l, (k, v) in enumerate(cache_kv(cache)):
        if hasattr(cache, "key_cache"):
            cache.key_cache[l] = k[:, :, :n_keep, :]
            cache.value_cache[l] = v[:, :, :n_keep, :]
        else:
            cache.layers[l].keys = k[:, :, :n_keep, :]
            cache.layers[l].values = v[:, :, :n_keep, :]
    return cache


def step(model, ids, cache):
    n = 0 if cache is None else cache.get_seq_length()
    am = torch.ones((1, n + ids.shape[1]), dtype=torch.long, device=ids.device)
    o = model(ids, attention_mask=am, past_key_values=cache, use_cache=True)
    return o.logits[0].float(), o.past_key_values


def sample(p):
    return int(torch.multinomial(p, 1))


# ------------------------------------------------------------------ Rebuilder
class Rebuilder:
    """From longrun.py / longrun_refill.py, unchanged: replaces k_proj/v_proj
    output for layers depth+1..NL-1 at the current position when armed."""

    def __init__(self, model):
        self.on = False
        self.depth = None
        self.hD = None
        self.NL = model.config.num_hidden_layers
        self.M = {}
        self.h = []
        L = model.model.layers

        def cap(i):
            def f(mod, args, kwargs):
                if self.on and i == self.depth:
                    x = kwargs.get("hidden_states", args[0] if args else None)
                    if x is not None:
                        self.hD = x[:, -1:, :].detach()
                return None
            return f

        def repl(i, w):
            def f(mod, inp, out):
                if (self.on and self.depth is not None and i > self.depth
                        and self.hD is not None):
                    o = out.clone()
                    P = self.M[(self.depth, i, w)]
                    o[:, -1:, :] = (self.hD.float() @ P[:-1, :]
                                    + P[-1, :]).to(o.dtype)
                    return o
                return out
            return f

        for i, l in enumerate(L):
            self.h.append(l.register_forward_pre_hook(cap(i), with_kwargs=True))
            self.h.append(l.self_attn.k_proj.register_forward_hook(repl(i, "K")))
            self.h.append(l.self_attn.v_proj.register_forward_hook(repl(i, "V")))

    def load(self, path, D, dev):
        z = np.load(path)
        for L in range(D + 1, self.NL):
            for w in ("K", "V"):
                k = f"M_{w}_{L}"
                if k in z:
                    self.M[(D, L, w)] = torch.tensor(z[k], dtype=torch.float32,
                                                     device=dev)
        n = sum(1 for k in self.M if k[0] == D)
        assert n == 2 * (self.NL - D - 1), f"expected full map set, got {n}"
        return n


# ---------------------------------------------------------------- generators
def plain_sample(model, tok, enc, seed, dev):
    torch.manual_seed(seed)
    ids, cache, out = enc["input_ids"], None, []
    with torch.no_grad():
        for _ in range(MAXNEW):
            lg, cache = step(model, ids, cache)
            t = sample(torch.softmax(lg[-1] / TAU, -1))
            out.append(t)
            if t in EOS:
                break
            ids = torch.tensor([[t]], device=dev)
    return out


def depth_sample(model, rb, tok, enc, seed, dev, D, refill_B):
    """Sampled early-exit generation with fitted repair map at depth D.
    Prompt is prefilled EXACTLY (hooks off); every generated position's
    upper-layer K/V come from the map. refill_B > 0: after each refill_B
    generated tokens, crop them from the cache and re-feed in one batched
    full-depth pass (hooks off), whose last-row logits produce the next token.
    refill_B = 0: no refill (accumulating damage, the -1.05 anchor)."""
    torch.manual_seed(seed)
    rb.on = False
    with torch.no_grad():
        lg, cache = step(model, enc["input_ids"], None)
        rb.depth, rb.on = D, True
        out, stale, n_refills = [], [], 0
        for _ in range(MAXNEW):
            t = sample(torch.softmax(lg[-1] / TAU, -1))
            out.append(t)
            if t in EOS:
                break
            lg, cache = step(model, torch.tensor([[t]], device=dev), cache)
            stale.append(t)
            if refill_B and len(stale) >= refill_B:
                crop_cache(cache, cache.get_seq_length() - len(stale))
                rb.on = False
                lg, cache = step(model, torch.tensor([stale], device=dev), cache)
                rb.on = True
                stale = []
                n_refills += 1
    rb.on = False
    return out, n_refills


def spec_sample(tgt, drf, tok, enc, seed, dev, lam):
    """Speculative loop from quant_ladder.py, plus per-window EAR.
    Returns (tokens, runs, ears)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        lgs_t, cache_t = step(tgt, enc["input_ids"], None)
        pend_t = torch.softmax(lgs_t[-1] / TAU, -1)
        lgs_d, cache_d = step(drf, enc["input_ids"], None)
        pend_d = torch.softmax(lgs_d[-1] / TAU, -1)
        emitted, unproc_t, unproc_d, runs, ears = [], [], [], [], []
        while len(emitted) < MAXNEW:
            if unproc_d:
                lg, cache_d = step(drf, torch.tensor([unproc_d], device=dev),
                                   cache_d)
                pend_d = torch.softmax(lg[-1] / TAU, -1)
                unproc_d = []
            draft_toks, draft_p = [], []
            p = pend_d
            for _ in range(B):
                t = sample(p)
                draft_toks.append(t)
                draft_p.append(p)
                lg, cache_d = step(drf, torch.tensor([[t]], device=dev),
                                   cache_d)
                p = torch.softmax(lg[-1] / TAU, -1)
            batch = unproc_t + draft_toks
            lgs, cache_t = step(tgt, torch.tensor([batch], device=dev),
                                cache_t)
            k = len(unproc_t)
            tdists = [pend_t if k == 0 else
                      torch.softmax(lgs[k - 1] / TAU, -1)]
            for j in range(1, B):
                tdists.append(torch.softmax(lgs[k + j - 1] / TAU, -1))
            bonus_dist = torch.softmax(lgs[k + B - 1] / TAU, -1)
            base_t = cache_t.get_seq_length() - B
            base_d = cache_d.get_seq_length() - B
            unproc_t = []
            # EAR over the window (independent of the sampled acceptance coin)
            ears.append(float(np.mean([float(torch.minimum(tdists[j],
                                                              draft_p[j]).sum())
                                       for j in range(B)])))
            acc = 0
            for j in range(B):
                x = draft_toks[j]
                pt, pd = tdists[j], draft_p[j]
                ratio = float(pt[x]) / max(float(pd[x]), 1e-12)
                if float(torch.rand(1)) < min(1.0, ratio / lam):
                    acc += 1
                    continue
                res = torch.clamp(pt - pd, min=0.0)
                s = float(res.sum())
                repl = sample(res / s) if s > 1e-12 else sample(pt)
                emitted += draft_toks[:acc] + [repl]
                crop_cache(cache_t, base_t + acc)
                crop_cache(cache_d, base_d + acc)
                unproc_t, unproc_d = [repl], [repl]
                pend_t = None
                break
            else:
                bonus = sample(bonus_dist)
                emitted += draft_toks + [bonus]
                unproc_t, unproc_d = [bonus], [bonus]
                pend_t = None
            runs.append(acc)
            if any(t in EOS for t in emitted[-B - 1:]):
                break
    cut = next((i for i, t in enumerate(emitted) if t in EOS), None)
    if cut is not None:
        emitted = emitted[:cut + 1]
    return emitted, runs, ears


# --------------------------------------------------------------------- setup
def load_models(dev):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    tok = AutoTokenizer.from_pretrained(MODELS[KEY])
    drf = drf3 = None
    from transformers import GenerationConfig
    try:
        ge = GenerationConfig.from_pretrained(MODELS[KEY]).eos_token_id
    except Exception:
        ge = None
    for e in ([ge] if isinstance(ge, int) else (ge or [])) + [tok.eos_token_id]:
        if e is not None:
            EOS.add(int(e))
    print(f"EOS ids: {sorted(EOS)} ({[tok.decode([e]) for e in sorted(EOS)]})",
          flush=True)
    if "raw3bit" in ARMS:
        # transformers' HqqConfig path is disabled in current versions; use the
        # hqq library directly: load bf16 on CPU, quantize layer-wise onto GPU.
        from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
        from hqq.models.hf.base import AutoHQQHFModel
        m0 = gb()
        drf3 = AutoModelForCausalLM.from_pretrained(
            MODELS[KEY], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        qc = BaseQuantizeConfig(nbits=HQQ_BITS, group_size=HQQ_GROUP, axis=1)
        AutoHQQHFModel.quantize_model(drf3, quant_config=qc,
                                      compute_dtype=torch.bfloat16, device=dev)
        HQQLinear.set_backend(HQQBackend.PYTORCH)
        drf3 = drf3.eval()
        print(f"HQQ {HQQ_BITS}-bit g{HQQ_GROUP} loaded: +{gb()-m0:.1f} GB "
              f"(NF4 is ~5.6; 3-bit should be smaller)", flush=True)
    tgt = AutoModelForCausalLM.from_pretrained(
        MODELS[KEY], torch_dtype=torch.bfloat16, device_map={"": 0}).eval()
    print(f"target bf16 loaded: total {gb():.1f} GB", flush=True)
    if any(a in ARMS for a in ("lam", "raw4bit")):
        m0 = gb()
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_use_double_quant=True)
        drf = AutoModelForCausalLM.from_pretrained(
            MODELS[KEY], quantization_config=qc, device_map={"": 0}).eval()
        print(f"NF4 loaded: +{gb()-m0:.1f} GB", flush=True)
    rb = None
    if any(a in ARMS for a in ("refill", "noref")):
        rb = Rebuilder(tgt)
        p = os.path.join(FITS, FITPAT.format(D=DEPTH))
        n = rb.load(p, DEPTH, dev)
        print(f"loaded {n} repair maps for depth {DEPTH} from {p}", flush=True)
    return tok, tgt, drf, drf3, rb


def load_domains():
    fetch()
    doms = {}
    for dom, fn in STD.items():
        if dom in NS:
            doms[dom] = [dict(prompt=p) for p in json.load(open(fn))[:NS[dom]]]
    if "hard" in NS:
        h = json.load(open(HARD))
        items = []
        for it in h["items"][:NS["hard"]]:
            suf = h["math_suffix"] if it["kind"] == "math" else h["code_suffix"]
            items.append(dict(prompt=it["prompt"] + suf, hard_id=it["id"],
                              kind=it["kind"], truth=it.get("truth"),
                              tolerance=it.get("tolerance"),
                              tests=it.get("tests")))
        doms["hard"] = items
    return doms


def teacher_forced_ear(tgt, other, tok, enc, ref_toks, dev):
    """Mean EAR of `other` vs `tgt` over a recorded reference continuation:
    quantization-applied check for the self-test."""
    ids = torch.cat([enc["input_ids"], torch.tensor([ref_toks], device=dev)], 1)
    with torch.no_grad():
        lt, _ = step(tgt, ids, None)
        lo, _ = step(other, ids, None)
    P = enc["input_ids"].shape[1]
    pt = torch.softmax(lt[P - 1:-1] / TAU, -1)
    po = torch.softmax(lo[P - 1:-1] / TAU, -1)
    return float(torch.minimum(pt, po).sum(-1).mean())


def selftest(tok, tgt, drf, drf3, rb, dev):
    prompt = "Explain why the sky is blue."
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(dev)
    ok = True
    ref = plain_sample(tgt, tok, enc, 1, dev)
    print(f"reference: {len(ref)} tokens")
    if drf is not None:
        e4 = teacher_forced_ear(tgt, drf, tok, enc, ref[:200], dev)
        print(f"EAR NF4 vs bf16 (teacher-forced, 200 pos): {e4:.3f} "
              f"(Aug 30: ~0.94) {'OK' if 0.88 < e4 < 0.99 else 'CHECK'}")
        ok &= 0.88 < e4 < 0.99
        # control: draft == target must accept every window
        em, runs, ears = spec_sample(tgt, tgt, tok, enc, 2, dev, 1.0)
        fa = float(np.mean([r == B for r in runs]))
        print(f"control draft=target: {len(runs)} windows, frac fully accepted "
              f"{fa:.3f}, mean EAR {np.mean(ears):.4f} "
              f"{'OK' if fa > 0.9 else 'FAIL'}")
        ok &= fa > 0.9
        em, runs, ears = spec_sample(tgt, drf, tok, enc, 3, dev, 1.0)
        print(f"lam1.0 NF4: mean accepted {np.mean(runs):.2f}/{B}, mean EAR "
              f"{np.mean(ears):.3f}, {len(em)} tokens")
    if drf3 is not None:
        e3 = teacher_forced_ear(tgt, drf3, tok, enc, ref[:200], dev)
        print(f"EAR {HQQ_BITS}-bit vs bf16: {e3:.3f} (expected ~0.80; must be "
              f"below NF4) {'OK' if 0.5 < e3 < 0.93 else 'CHECK'}")
        ok &= 0.5 < e3 < 0.93
        r3 = plain_sample(drf3, tok, enc, 4, dev)
        print(f"raw3bit sample: {len(r3)} tokens; first 120 chars: "
              f"{tok.decode(r3[:60], skip_special_tokens=True)[:120]!r}")
    if rb is not None:
        # no-op check: hooks never armed, refill path reproduces plain sampling
        a = plain_sample(tgt, tok, enc, 5, dev)
        b, nr = depth_sample_noop(tgt, tok, enc, 5, dev)
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                   min(len(a), len(b)))
        print(f"refill no-op (hooks off) vs plain: divergence at {div} of "
              f"{len(a)} ({div/max(len(a),1):.0%}), {nr} refills "
              f"{'OK' if div/max(len(a),1) > 0.9 else 'MARGINAL'}")
        em, nr = depth_sample(tgt, rb, tok, enc, 6, dev, DEPTH, REFILL_B)
        print(f"d{DEPTH}_B{REFILL_B}: {len(em)} tokens, {nr} refills; "
              f"{tok.decode(em[:60], skip_special_tokens=True)[:120]!r}")
        em, nr = depth_sample(tgt, rb, tok, enc, 7, dev, DEPTH, 0)
        print(f"d{DEPTH}_noref: {len(em)} tokens; "
              f"{tok.decode(em[:60], skip_special_tokens=True)[:120]!r}")
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED -- stop and report'}")


def depth_sample_noop(model, tok, enc, seed, dev):
    """Refill mechanics with hooks off: crop-and-refeed must be an identity on
    the sampled sequence (same seed => same tokens, up to bf16 near-ties)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        lg, cache = step(model, enc["input_ids"], None)
        out, stale, n_refills = [], [], 0
        for _ in range(MAXNEW):
            t = sample(torch.softmax(lg[-1] / TAU, -1))
            out.append(t)
            if t in EOS:
                break
            lg, cache = step(model, torch.tensor([[t]], device=dev), cache)
            stale.append(t)
            if len(stale) >= REFILL_B:
                crop_cache(cache, cache.get_seq_length() - len(stale))
                lg, cache = step(model, torch.tensor([stale], device=dev), cache)
                stale = []
                n_refills += 1
    return out, n_refills


# ---------------------------------------------------------------------- main
def main():
    assert torch.cuda.is_available(), "pod script"
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)
    tok, tgt, drf, drf3, rb = load_models(dev)
    if SELFTEST:
        selftest(tok, tgt, drf, drf3, rb, dev)
        return
    doms = load_domains()
    print("domains:", {d: len(v) for d, v in doms.items()}, "arms:", ARMS,
          flush=True)

    t0 = time.time()
    total = sum(len(v) for v in doms.values())
    for dom, plist in doms.items():
        for qi, meta in enumerate(plist):
            path = f"{OUT}/{dom}_{qi:03d}.json"
            if os.path.exists(path):
                continue
            prompt = meta["prompt"]
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
                return_dict=True).to(dev)
            base = SEED * 100000 + DOM_IDX[dom] * 10000 + qi * 20
            pints = (prompt_ints(prompt) if dom == "arith"
                     or meta.get("kind") == "math" else set())
            items = []

            def rec(arm, toks, sec, extra=None):
                txt = tok.decode(toks, skip_special_tokens=True)
                d = dict(domain=dom, prompt_idx=qi, prompt=prompt, arm=arm,
                         depth=None, method=arm, text=txt, n_tokens=len(toks),
                         temperature=TAU, hit_cap=int(len(toks) >= MAXNEW),
                         rep8=round(rep8(txt), 4), final_ans=parse_final(txt),
                         ints_ok=(all(x in txt for x in pints)
                                  if pints else None),
                         sec=round(sec, 1))
                for k in ("hard_id", "kind", "truth", "tolerance"):
                    if k in meta:
                        d[k] = meta[k]
                if extra:
                    d.update(extra)
                items.append(d)

            if "reference" in ARMS:
                ts = time.time(); r = plain_sample(tgt, tok, enc, base + 0, dev)
                rec("reference", r, time.time() - ts)
            if "resample" in ARMS:
                ts = time.time(); r = plain_sample(tgt, tok, enc, base + 1, dev)
                rec("resample", r, time.time() - ts)
            if "lam" in ARMS:
                for li, lam in enumerate(LAMS):
                    ts = time.time()
                    em, runs, ears = spec_sample(tgt, drf, tok, enc,
                                                 base + 2 + li, dev, lam)
                    rec(f"lam{lam}", em, time.time() - ts,
                        dict(lam=lam, n_windows=len(runs),
                             mean_accept=round(float(np.mean(runs)), 3),
                             frac_full=round(float(np.mean(
                                 [a == B for a in runs])), 3),
                             ear_mean=round(float(np.mean(ears)), 4)))
            if "speccontrol" in ARMS:
                # implementation control: draft == target, strict rule.
                # Differs from plain sampling only in the loop (batched verify,
                # cache cropping, ride-along tokens).
                ts = time.time()
                em, runs, ears = spec_sample(tgt, tgt, tok, enc, base + 5, dev, 1.0)
                rec("speccontrol", em, time.time() - ts,
                    dict(lam=1.0, n_windows=len(runs),
                         mean_accept=round(float(np.mean(runs)), 3),
                         frac_full=round(float(np.mean([a == B for a in runs])), 3),
                         ear_mean=round(float(np.mean(ears)), 4)))
            if "raw4bit" in ARMS:
                ts = time.time(); r = plain_sample(drf, tok, enc, base + 6, dev)
                rec("raw4bit", r, time.time() - ts)
            if "raw3bit" in ARMS:
                ts = time.time(); r = plain_sample(drf3, tok, enc, base + 7, dev)
                rec("raw3bit", r, time.time() - ts)
            if "refill" in ARMS:
                ts = time.time()
                em, nr = depth_sample(tgt, rb, tok, enc, base + 8, dev, DEPTH,
                                      REFILL_B)
                rec(f"d{DEPTH}_B{REFILL_B}", em, time.time() - ts,
                    dict(depth=DEPTH, refill_B=REFILL_B, n_refills=nr))
            if "noref" in ARMS:
                ts = time.time()
                em, nr = depth_sample(tgt, rb, tok, enc, base + 9, dev, DEPTH, 0)
                rec(f"d{DEPTH}_noref", em, time.time() - ts, dict(depth=DEPTH))
            json.dump(items, open(path, "w"), ensure_ascii=False)
            done = len([f for f in os.listdir(OUT) if f.endswith(".json")])
            el = time.time() - t0
            per = el / max(1, done)
            print(f"{dom} q{qi:03d}: ref {items[0]['n_tokens']} tok; "
                  + " ".join(f"{i['arm']}={i['sec']:.0f}s" for i in items)
                  + f"  [{done}/{total}, eta {(total-done)*per/60:4.0f} min]",
                  flush=True)

    allr = []
    for f in sorted(os.listdir(OUT)):
        if f.endswith(".json"):
            d = json.load(open(f"{OUT}/{f}"))
            if isinstance(d, list):
                allr += d
    json.dump(allr, open(ITEMS_OUT, "w"), ensure_ascii=False)
    arms = sorted({r["arm"] for r in allr})
    print(f"\nwrote {ITEMS_OUT}: {len(allr)} records; arms {arms}")

    print("\nWALL SECONDS PER ARM (mean):")
    for a in arms:
        g = [r["sec"] for r in allr if r["arm"] == a]
        print(f"  {a:>12}: {np.mean(g):6.1f}s  n={len(g)}")
    print("\nACCEPTANCE / EAR BY ARM AND DOMAIN (speculative arms):")
    for a in arms:
        g = [r for r in allr if r["arm"] == a and "ear_mean" in r]
        if not g:
            continue
        row = f"  {a:>8}: "
        for dom in doms:
            gg = [r for r in g if r["domain"] == dom]
            if gg:
                row += (f"{dom} acc {np.mean([r['mean_accept'] for r in gg]):4.2f}"
                        f"/EAR {np.mean([r['ear_mean'] for r in gg]):.3f}  ")
        print(row)
    print("\nSURFACE: hit_cap / rep8>0.1 / ints_ok by arm:")
    for a in arms:
        g = [r for r in allr if r["arm"] == a]
        io = [r["ints_ok"] for r in g if r["ints_ok"] is not None]
        print(f"  {a:>12}: cap {np.mean([r['hit_cap'] for r in g]):.2f}  "
              f"rep {np.mean([r['rep8'] > 0.1 for r in g]):.2f}  "
              f"ints_ok {np.mean(io) if io else float('nan'):.2f}")
    print("\nHARD-MATH FINAL ANSWER vs TRUTH (fraction correct, by arm):")
    for a in arms:
        g = [r for r in allr if r["arm"] == a and r.get("kind") == "math"]
        if not g:
            continue
        c = 0
        for r in g:
            try:
                fa = r["final_ans"]
                val = (float(fa.split("/")[0]) / float(fa.split("/")[1])
                       if "/" in fa else float(fa))
                tr = r["truth"]
                trv = (float(str(tr).split("/")[0]) / float(str(tr).split("/")[1])
                       if isinstance(tr, str) and "/" in tr else float(tr))
                tol = r.get("tolerance") or 1e-6
                c += abs(val - trv) <= tol
            except Exception:
                pass
        print(f"  {a:>12}: {c}/{len(g)}")


if __name__ == "__main__":
    main()

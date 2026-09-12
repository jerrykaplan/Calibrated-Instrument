"""Item-level EAR for the raw quantized arms (pod).

For every record in ITEMS with arm in {raw4bit, raw3bit, reference}, re-tokenize
prompt + response, run ONE batched forward through bf16 and through the
matching quantized model(s), and record

  ear_mean   mean over generated positions of sum_k min(p_bf16(k), p_q(k))
             at tau=TAU  (expected acceptance rate = 1 - TV distance)
  top1_agree fraction of generated positions where argmax agrees

raw4bit items are scored against NF4; raw3bit against HQQ 3-bit; reference
items against both (the "same prefix" EAR, for comparison with the
along-own-path number). Output: ce_ear_<KEY>.json keyed by uid.

  KEY=q7 ITEMS=ce_items.json HF_HOME=/workspace/hf python3 ce_ear.py
  KEY=l8 ITEMS=ce_items_l8.json HF_HOME=/workspace/hf python3 ce_ear.py
"""
import json, os, time
import numpy as np
import torch

KEY = os.environ.get("KEY", "q7")
ITEMS = os.environ.get("ITEMS", "ce_items.json")
OUT = os.environ.get("OUT", f"ce_ear_{KEY}.json")
TAU = float(os.environ.get("TAU", "0.7"))
HQQ_BITS = int(os.environ.get("HQQ_BITS", "3"))
HQQ_GROUP = int(os.environ.get("HQQ_GROUP", "64"))
MODELS = {"q7": "Qwen/Qwen2.5-7B-Instruct", "l8": "meta-llama/Llama-3.1-8B-Instruct"}


def gb():
    return torch.cuda.memory_allocated() / 1e9


def load_models(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear, HQQBackend
    from hqq.models.hf.base import AutoHQQHFModel
    tok = AutoTokenizer.from_pretrained(MODELS[KEY])
    m3 = AutoModelForCausalLM.from_pretrained(MODELS[KEY], torch_dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True)
    AutoHQQHFModel.quantize_model(m3, quant_config=BaseQuantizeConfig(
        nbits=HQQ_BITS, group_size=HQQ_GROUP, axis=1),
        compute_dtype=torch.bfloat16, device=dev)
    HQQLinear.set_backend(HQQBackend.PYTORCH)
    m3 = m3.eval()
    print(f"HQQ {HQQ_BITS}-bit loaded: {gb():.1f} GB", flush=True)
    m16 = AutoModelForCausalLM.from_pretrained(MODELS[KEY], torch_dtype=torch.bfloat16,
                                               device_map={"": 0}).eval()
    print(f"bf16 loaded: total {gb():.1f} GB", flush=True)
    m4 = AutoModelForCausalLM.from_pretrained(MODELS[KEY], quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True), device_map={"": 0}).eval()
    print(f"NF4 loaded: total {gb():.1f} GB", flush=True)
    return tok, m16, m4, m3


@torch.no_grad()
def logits_resp(model, ids, P):
    """Logits (bf16, [L-P, V]) at positions P-1 .. L-2, i.e. the positions
    that generate the response tokens. One forward, no cache."""
    lg = model(ids, use_cache=False).logits[0]
    return lg[P - 1:-1].to(torch.bfloat16).clone()


@torch.no_grad()
def ear_pair(lg16, lgq, block=256):
    """Mean EAR and top-1 agreement over positions, computed in position
    blocks so that only a block of float32 distributions exists at once."""
    n = lg16.shape[0]
    ear_sum, agree = 0.0, 0
    for s in range(0, n, block):
        p16 = torch.softmax(lg16[s:s + block].float() / TAU, -1)
        pq = torch.softmax(lgq[s:s + block].float() / TAU, -1)
        ear_sum += float(torch.minimum(p16, pq).sum(-1).sum())
        agree += int((p16.argmax(-1) == pq.argmax(-1)).sum())
        del p16, pq
    return ear_sum / n, agree / n


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    tok, m16, m4, m3 = load_models(dev)
    items = [r for r in json.load(open(ITEMS)) if r["arm"] in ("raw4bit", "raw3bit", "reference")]
    print(f"{len(items)} items", flush=True)
    out = {}
    if os.path.exists(OUT):
        out = json.load(open(OUT))
        print(f"resuming with {len(out)} done", flush=True)
    t0 = time.time()
    for n, r in enumerate(items):
        uid = f"{r['domain']}_{r['prompt_idx']}_{r['arm']}"
        if uid in out:
            continue
        enc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(dev)
        resp = tok(r["text"], add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        ids = torch.cat([enc["input_ids"], resp], 1)
        P = enc["input_ids"].shape[1]
        if resp.shape[1] < 2:
            continue
        lg16 = logits_resp(m16, ids, P)
        rec = dict(arm=r["arm"], domain=r["domain"], prompt_idx=r["prompt_idx"],
                   n_resp_tokens=int(resp.shape[1]))
        quants = {"raw4bit": [("nf4", m4)], "raw3bit": [("hqq3", m3)],
                  "reference": [("nf4", m4), ("hqq3", m3)]}[r["arm"]]
        for name, m in quants:
            lgq = logits_resp(m, ids, P)
            e, t1 = ear_pair(lg16, lgq)
            rec[f"ear_{name}"] = round(e, 4)
            rec[f"top1_{name}"] = round(t1, 4)
            del lgq
        del lg16
        torch.cuda.empty_cache()
        out[uid] = rec
        if (n + 1) % 20 == 0:
            json.dump(out, open(OUT, "w"))
            print(f"  {n+1}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {OUT}: {len(out)} records")
    for arm, key in (("raw4bit", "ear_nf4"), ("raw3bit", "ear_hqq3"),
                     ("reference", "ear_nf4"), ("reference", "ear_hqq3")):
        g = [v for v in out.values() if v["arm"] == arm and key in v]
        if g:
            row = f"  {arm:>9} {key:>9}: "
            for d in ("prose", "code", "arith", "zh", "hard"):
                gg = [v[key] for v in g if v["domain"] == d]
                if gg:
                    row += f"{d} {np.mean(gg):.3f}  "
            print(row)


if __name__ == "__main__":
    main()

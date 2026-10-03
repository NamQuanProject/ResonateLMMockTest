"""Convert TinyLlama-1.1B attention layers to ResonatorLM mixers and test the idea.

Stage 1  layer-wise attention transfer: each mixer regresses its attention layer's output
Stage 2  end-to-end distillation: KL(teacher logits || student logits), mixers only trainable
Then:    WikiText-2 perplexity, passkey retrieval, text samples, conv-vs-recurrence check.

    python run_1b.py --variant adjusted --stage1 300 --stage2 150
    python run_1b.py --variant teacher            # unmodified TinyLlama reference
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_resonator import (ResonatorConfig, convert, load_student_state_dict, set_mode,
                             student_parameters, student_state_dict, switches)

MODEL = "TinyLlama/TinyLlama_v1.1"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = Path(__file__).parent / "runs"

PROMPTS = [
    "The capital of France is",
    "In machine learning, overfitting happens when",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Software testing is important because",
]


def token_stream(tok, split):
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    return torch.tensor(tok(text).input_ids)


def batches(ids, seq, bs, seed=0):
    g = random.Random(seed)
    while True:
        starts = [g.randrange(0, len(ids) - seq - 1) for _ in range(bs)]
        yield torch.stack([ids[s:s + seq] for s in starts]).to(DEV)


@torch.no_grad()
def perplexity(model, ids, seq=1024, n_chunks=24):
    nll, n = 0.0, 0
    for i in range(n_chunks):
        x = ids[i * seq:(i + 1) * seq][None].to(DEV)
        logits = model(x, use_cache=False).logits.float()
        nll += F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="sum").item()
        n += seq - 1
    return math.exp(nll / n)


@torch.no_grad()
def passkey(model, tok, fillers=(5, 20, 60), trials=10):
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    res = {}
    rng = random.Random(1)
    for k in fillers:
        hits, ntok = 0, 0
        for _ in range(trials):
            key = str(rng.randrange(10000, 99999))
            p = (f"There is an important pass key hidden inside a lot of irrelevant text. Find it and memorize it. "
                 f"The pass key is {key}. Remember it. {key} is the pass key. " + filler * k +
                 "What is the pass key? The pass key is")
            x = tok(p, return_tensors="pt").input_ids.to(DEV)
            ntok = x.shape[1]
            out = model.generate(x, max_new_tokens=8, do_sample=False, use_cache=False,
                                 pad_token_id=tok.eos_token_id)
            hits += key in tok.decode(out[0, ntok:])
        res[f"{ntok}_tokens"] = hits / trials
    return res


@torch.no_grad()
def samples(model, tok, n=40):
    outs = []
    for p in PROMPTS:
        x = tok(p, return_tensors="pt").input_ids.to(DEV)
        y = model.generate(x, max_new_tokens=n, do_sample=False, use_cache=False, repetition_penalty=1.1,
                           pad_token_id=tok.eos_token_id)
        outs.append({"prompt": p, "completion": tok.decode(y[0, x.shape[1]:], skip_special_tokens=True)})
    return outs


@torch.no_grad()
def recurrence_check(model, tok):
    """Greedy decode with the recurrent state (use_cache=True) vs re-running the FFT path each step."""
    x = tok(PROMPTS[1], return_tensors="pt").input_ids.to(DEV)
    a = model.generate(x, max_new_tokens=24, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
    set_mode(model, "student")
    b = model.generate(x, max_new_tokens=24, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
    set_mode(model, "student")
    logits_full = model(b, use_cache=False).logits[0, -1].float()
    set_mode(model, "student")
    pre = model(b[:, :-1], use_cache=True)                  # prefill all but last token
    # Resonator layers never write the KV cache, so HF infers position 0 when layer 0 is a resonator;
    # pass the real position so hybrid attention layers get the right RoPE angle and mask.
    pos = torch.tensor([[b.shape[1] - 1]], device=b.device)
    logits_step = model(b[:, -1:], past_key_values=pre.past_key_values, use_cache=True,
                        position_ids=pos, cache_position=pos[0]).logits[0, -1].float()
    set_mode(model, "student")
    rel = ((logits_full - logits_step).abs().max() / logits_full.abs().max()).item()
    return {"same_greedy_tokens": bool(torch.equal(a, b)), "last_logit_rel_err": rel}


def train(model, ids, args, log):
    params = student_parameters(model)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = args.stage1 + args.stage2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / max(1, total))))
    data = batches(ids, args.seq, args.bs)
    ckpt = OUT / f"{args.variant}_ckpt.pt"
    start = 0
    if ckpt.exists():
        c = torch.load(ckpt)
        load_student_state_dict(model, c["mixers"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        start = c["step"]
        log.extend(c["log"])
        print(f"resumed from step {start}", flush=True)
    for _ in range(start):
        next(data)                              # same batch order as an uninterrupted run
    t0 = time.time()
    for step in range(start, total):
        x = next(data)
        if step < args.stage1:
            set_mode(model, "distill")
            model(x, use_cache=False)
            loss = torch.stack([s.loss for s in switches(model)]).mean()
        else:
            with torch.no_grad():
                set_mode(model, "teacher")
                t = model(x, use_cache=False).logits.float()
            set_mode(model, "student")
            s = model(x, use_cache=False).logits.float()
            loss = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="batchmean") / x.shape[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if step % 10 == 0 or step == total - 1:
            stage = "attn-transfer" if step < args.stage1 else "kd"
            msg = f"step {step:4d} [{stage}] loss {loss.item():.4f}  {time.time() - t0:6.0f}s"
            print(msg, flush=True); log.append(msg)
        if (step + 1) % 50 == 0 or step == total - 1:
            OUT.mkdir(exist_ok=True)
            torch.save({"mixers": student_state_dict(model), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step + 1, "log": log}, ckpt)
    set_mode(model, "student")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="adjusted", choices=["teacher", "report", "adjusted", "hybrid"])
    ap.add_argument("--attn_layers", type=int, nargs="*", default=None, help="hybrid: layers that keep attention")
    ap.add_argument("--stage1", type=int, default=300)
    ap.add_argument("--stage2", type=int, default=150)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--load", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    train_ids, test_ids = token_stream(tok, "train"), token_stream(tok, "test")
    name = args.variant
    result, log = {"variant": name, "args": vars(args)}, []

    if name != "teacher":
        kw = {"attn_layers": args.attn_layers} if args.attn_layers is not None else {}
        cfg = ResonatorConfig.variant(name, **kw)
        convert(model, cfg)
        n_train = sum(p.numel() for p in student_parameters(model))
        result.update(converted_layers=len(switches(model)), attn_layers=cfg.attn_layers,
                      trainable_params=n_train)
        print(f"{name}: {len(switches(model))} resonator layers, {n_train / 1e6:.1f}M trainable params")
        set_mode(model, "student")
        result["ppl_before_training"] = perplexity(model, test_ids, n_chunks=8)
        print("ppl before training", result["ppl_before_training"])
        if args.load:
            load_student_state_dict(model, torch.load(args.load))
        else:
            train(model, train_ids, args, log)
            OUT.mkdir(exist_ok=True)
            torch.save(student_state_dict(model), OUT / f"{name}_mixers.pt")
        set_mode(model, "student")
        result["recurrence_check"] = recurrence_check(model, tok)
        print("recurrence", result["recurrence_check"])
        # Decoding state vs KV cache, per sequence (bf16 KV, fp32 resonator state).
        c = model.config
        kv_per_tok = 2 * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * 2
        n_attn = c.num_hidden_layers - len(switches(model))
        res_state = sum(s.student.state_bytes() for s in switches(model))
        result["decode_memory_1M_tokens_GiB"] = (n_attn * kv_per_tok * 1_000_000 + res_state) / 2**30
        result["decode_memory_teacher_1M_tokens_GiB"] = c.num_hidden_layers * kv_per_tok * 1_000_000 / 2**30

    result["ppl"] = perplexity(model, test_ids)
    print("ppl", result["ppl"])
    result["passkey"] = passkey(model, tok)
    print("passkey", result["passkey"])
    result["samples"] = samples(model, tok)
    for s in result["samples"]:
        print(f"--- {s['prompt']!r}\n{s['completion']}")
    result["log"] = log
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

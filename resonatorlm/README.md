# ResonatorLM on a pretrained 1B model (TinyLlama-1.1B)

Rebuild of the ResonatorLM mixer from `../katalon_report.pdf` (Sec. 2). The rebuilt mixer replaces the
attention layers of pretrained **TinyLlama-1.1B**, and the rest of the model (embeddings, RMSNorm, SwiGLU MLPs, LM head)
keeps its pretrained weights, frozen. Only the mixers are trained (95–116M params), by distillation from the
original model on WikiText-2 train (450 steps × 2 × 512 tokens ≈ 0.46M tokens, Apple M2, ~35 min per variant).

| file | what |
|---|---|
| `resonator.py` | mixer: damped-oscillator kernel (eq. 4), causal FFT conv (eq. 5/10), complex recurrence (eq. 6–7), gate + local conv (eq. 8), head coupling (eq. 13) |
| `llama_resonator.py` | swaps mixers into a Llama model; drive ← pretrained `v_proj` (GQA-expanded), output ← `o_proj` |
| `run_1b.py` | stage 1 layer-wise attention transfer → stage 2 logit KD → perplexity, passkey, samples, recurrence check |
| `run_all.sh` | runs all variants sequentially (resumable) |

## Variants

| | report | adjusted | hybrid |
|---|---|---|---|
| timescales | 1 per head (32) | 1 per channel (2048) | per channel |
| drive gain | raw kernel (eq. 5) | × (1−e^{−α}), unit DC gain | same |
| content-dependent input gate | – | yes (kernel stays LTI) | yes |
| head coupling / local conv | on | off (report ablation: no gain) | off |
| attention kept | none | none | layers 5, 10, 15, 20 |

## Results (WikiText-2 test, 24×1024 tokens; passkey = 10 trials per length)

| model | PPL | passkey @177 / 537 / 1497 tok | decode memory @ 1M tokens |
|---|---|---|---|
| TinyLlama-1.1B (original) | **8.50** | 100% / 100% / 100% | 21.0 GiB KV cache |
| report mixer, all 22 layers | 64.0 | 0 / 0 / 0 | 0.9 MB state |
| adjusted mixer, all 22 layers | 28.4 | 0 / 0 / 0 | 0.35 MB state |
| hybrid (18 resonator + 4 attention) | 18.5 | 0 / 0 / 0 | 3.8 GiB |

Perplexity right after conversion, before training: 236 591 (report), 4 203 (adjusted), 2 933 (hybrid).

Correctness:
- Mixer-level checks pass in fp32. Causality error is 4e-8 (adjusted) and 1.2e-6 (report, unnormalised gain). FFT and recurrence agree to ~1e-6 relative.
- In the full model, FFT and recurrent decoding agree to 1e-7 in fp64. In bf16 on MPS, adjusted and hybrid give the same greedy tokens. The report variant drifts after a near-tie, with last-logit error 0.5%.

## What this says about the idea

1. **The efficiency claim holds.** Decode memory is a fixed state: 0.35 MB versus a 21 GiB KV cache at 1M tokens, with exact conv/recurrence equivalence.
2. **Quality does not transfer cheaply.** Replacing all attention costs 3.3× perplexity (8.5 → 28.4) after 0.46M distillation tokens. A layer-wise transfer loss of ~0.45 means each mixer reproduces only about 55% of its attention layer's output.
3. **Retrieval is lost entirely, even in the hybrid.** The report lists exact retrieval as unverified (Sec. 2.3, 3.5). Here every resonator model drops from 100% to 0% passkey accuracy, even at 177 tokens. Fluent but generic continuations ("The capital of France is the most important city in the world") are the same symptom: an LTI kernel blurs token identity.
4. **The adjustments matter at 1B scale.** The adjusted mixer gets 28.4 PPL versus 64.0 for the report mixer. The report mixer's attention-transfer loss started at ~21,800 (relative MSE), because its unnormalised kernel sums inputs over half-lives of up to 2048 tokens. Four changes were made together, so this run cannot say which one helps most.

Caveats: this is short distillation on one small dataset, one seed, and token-level BPE perplexity, which is not comparable with the report's character-level numbers. The base model gives odd greedy answers to short prompts even unmodified (it predicts "." after "The capital of France is"), so compare samples across variants, not against expectations.

## Reproduce

```bash
source ../.venv/bin/activate
./run_all.sh                                    # all variants, ~2.5 h on an M2
python run_1b.py --variant adjusted --load runs/adjusted_mixers.pt   # re-evaluate only
```

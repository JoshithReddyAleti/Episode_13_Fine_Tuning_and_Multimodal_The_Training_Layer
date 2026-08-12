# PEFT Method Comparison

| Method | Params | VRAM (7B) | Time | Quality | Notes |
|---|---|---|---|---|---|
| Full FT | 7B | 60 GB | 1× | Baseline | Rarely used in prod |
| LoRA r=16 | ~30M | 16 GB | 0.5× | 97% | Modern default |
| QLoRA r=16 | ~30M | 8 GB | 0.7× | 96% | Default for 65B+ single-GPU |
| DoRA r=16 | ~32M | 18 GB | 0.6× | 98% | +0.5-2% over LoRA, ~20% slower |
| LongLoRA | varies | varies | varies | varies | Only for context extension |
| AdaLoRA | ~30M | 17 GB | 0.7× | 95-97% | Adaptive rank per layer |
| IA³ | ~1M | 15 GB | 0.4× | 93% | Extreme efficiency, simple tasks |
| Prompt Tuning | ~50K | 14 GB | 0.3× | 85% | Big models only |
| Prefix Tuning | ~200K | 15 GB | 0.3× | 85-90% | Superseded by LoRA |

## Decision tree

1. Base fits FP16/BF16? → LoRA.
2. Doesn't fit? → QLoRA.
3. Want +1-2% quality? → DoRA (accept 20% slower).
4. Need longer context? → LongLoRA.
5. Extreme memory constraint + simple task? → IA³.

## Target modules

- **Attention-only** (`q_proj`, `v_proj`) — light, ~0.1% params.
- **Attention full** (`q, k, v, o`) — ~0.3%.
- **Attention + MLP** (all linear) — **modern default**, ~0.7%.
- **All linear** — max quality, ~1%.

## Rank guidance

- Simple style: r=4-8.
- Instruction following: r=8-16.
- Complex task: r=16-32.
- Approaching full FT: r=32-64.
- CPT substitute: r=128+ (or full FT).

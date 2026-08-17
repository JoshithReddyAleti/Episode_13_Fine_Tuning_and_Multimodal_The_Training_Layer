# 🪶 Parameter-Efficient Fine-Tuning — The Math, Code, and Choices

> *Full fine-tuning of a 70B model updates 140 GB of weights. LoRA updates 40 MB — 3,500× less — and reaches 95-99% of full FT quality on most tasks. This is not a compromise; it's the default in production.*

---

## Full Fine-Tuning vs PEFT (`full_finetune_vs_peft.py`)

Full fine-tuning updates every parameter. For 7B in FP16: ~60 GB VRAM in training. For 70B: ~600 GB, multi-node.

PEFT freezes the base and trains a small number of additional/modified parameters. Costs plummet.

| Aspect | Full FT | PEFT (LoRA-family) |
|---|---|---|
| Trainable params | 100% | 0.01-1% |
| VRAM (7B) | 60-100 GB | 12-24 GB |
| VRAM (70B) | Multi-node | 40-80 GB (QLoRA) |
| Training time | Baseline | 40-70% of full FT |
| Task-specific quality | Baseline | 95-99% |
| Multi-task/foundational | Baseline | 80-95% (larger gap) |
| Adapter size | Full model | 5-100 MB |
| Multi-tenant serving | 1 base per FT | 1 base + many adapters |
| Catastrophic forgetting | Higher risk | Lower risk |

**Prefer full FT when:** continued pretraining, base model must substantially change, quality matters more than cost, no need for multiple variants.

**Prefer PEFT when:** task-specific, multi-tenant, limited compute, preserving base capabilities matters, iterating quickly.

**In production 2026, PEFT (specifically LoRA/QLoRA) is the default.** Full FT is the exception.

---

## LoRA Deep Dive (`lora_deep_dive.py`)

**LoRA (Hu et al. 2021)** — the paper that made PEFT dominant.

**Insight:** the *update* to a pretrained weight matrix likely has low intrinsic rank. Approximate ΔW (`d×k`) as product of two much smaller matrices.

**Math:**
- Frozen weight: `W ∈ ℝ^(d×k)`.
- Learned update: `ΔW = B·A` where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×k)`, `r << min(d, k)`.
- Effective weight: `W + α·(B·A) / r`.
- Trainable params: `d·r + r·k` instead of `d·k`.

For 4096×4096 with r=8: 65K params vs 16.8M. **259× fewer.**

**Initialization:** A ~ Gaussian, B = 0. So ΔW = 0 at start — model behaves identically to base. Training gradually shifts.

**Merging:** `W_final = W + α·B·A / r`. Once merged, zero inference overhead.

**Serving without merging:** compute `Wx + (α/r)·B·(A·x)` at inference. Small overhead but swap adapters per-request — the multi-LoRA pattern.

**Typical hyperparameters:** r 4-64 (8-16 common); α often 2×r or fixed at 16/32; dropout 0.05-0.1.

---

## LoRA Rank Selection (`lora_rank_selection.py`)

**Rank r is the single most impactful LoRA hyperparameter.**

**By task type:**
- Simple style/format: r=4-8.
- Instruction following: r=8-16.
- Complex task behavior: r=16-32.
- Approaching full FT: r=32-64.
- CPT substitute: r=128+ (or full FT).

**How to choose:**
1. Start r=8.
2. Evaluate.
3. Underfit (both losses high) → increase r.
4. Overfit (train low, val high) → more data or lower r.
5. Diminishing returns around r=32-64.

**Rank vs α:** effective LR scales as α/r. Common: fix α=2r so effective scaling is constant when r changes. Some (Rank-Stabilized) use α=2√r.

---

## LoRA Target Modules (`lora_target_modules.py`)

**Which linear projections to adapt?**

Every transformer layer has: attention (q_proj, k_proj, v_proj, o_proj), MLP (gate_proj, up_proj, down_proj), embedding.

**Common configurations:**

**Attention-only (light):** `["q_proj", "v_proj"]`. Original LoRA paper's choice. ~0.1% params.

**Attention full:** `["q_proj", "k_proj", "v_proj", "o_proj"]`. ~0.3%. Good for instruction following.

**Attention + MLP (modern default):** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`. ~0.7%. **The pragmatic default.**

**All linear:** everywhere. ~1%. Maximum quality within LoRA constraints.

Guidance: attention+MLP is the pragmatic default. MLP contains substantial task-relevant computation.

---

## QLoRA Deep Dive (`qlora_deep_dive.py`)

**QLoRA (Dettmers et al. 2023)** — made 65B fine-tuning possible on one 48GB GPU.

**Insight:** LoRA avoids updating base weights. So base weights only need to be *readable*. If readable in 4-bit precision, no FP16 needed for them — LoRA adapter stays in FP16.

**Three key techniques:**

**1. 4-bit NormalFloat (NF4).** New 4-bit type optimized for normally-distributed weights. Loses less quality than naive INT4.

**2. Double quantization.** 4-bit weights need quantization constants (scales, zero points). Quantize those too. Saves another ~0.4 bits/param.

**3. Paged optimizers.** Optimizer states (Adam's m, v) are large. Under memory pressure, page to CPU on-the-fly.

**Result:** Llama 65B on one 48GB GPU. Quality within 1-2% of full FT. Training 30-50% longer (on-the-fly dequant).

**The default for production fine-tuning above 7B on constrained hardware.** Frameworks: `bitsandbytes` + `peft` + `transformers`, or Unsloth.

---

## DoRA (`dora.py`)

**DoRA (Liu et al. 2024)** — LoRA improvement decomposing weights into magnitude and direction.

**Insight:** weight update in fine-tuning decomposes into magnitude change and direction change. Full FT changes both; LoRA changes both entangled. DoRA parameterizes separately so dynamics better match full FT.

**Impact:** typically 0.5-2% quality improvement over LoRA at same rank, ~20% training overhead.

Available in `peft`. Prefer over LoRA when you want the extra quality and can afford the overhead.

---

## LongLoRA (`longlora.py`)

**Chen et al. 2023.** LoRA optimized for extending context length.

**Problem:** naive fine-tuning to extend context (4K → 32K) is expensive due to O(N²) attention.

**LongLoRA:**
- **Shifted sparse attention (S²-Attn)** during training: groups tokens, computes attention locally within groups, shifted across layers to still capture long-range.
- **LoRA on embedding and normalization** layers too — those matter more for context extension.

**Result:** extend Llama-7B from 4K → 100K on a single GPU with LoRA.

**When to use:** task requires longer context than base supports, and RoPE scaling alone isn't enough.

---

## AdaLoRA (`adalora.py`)

**Zhang et al. 2023.** Dynamically allocate rank across layers.

**Mechanism:** parameterize LoRA as SVD (A·Σ·B where Σ is singular values). Learn Σ during training; prune small ones. Different layers end up with different effective ranks.

**Advantage:** automatic per-layer rank allocation.

**Disadvantage:** more complex; slower convergence; doesn't always outperform well-tuned fixed-rank LoRA.

---

## IA³ (`ia3.py`)

**Liu et al. 2022.** Rather than low-rank matrices, learns per-channel *scaling vectors* that multiply activations element-wise.

**Params:** ~0.01% of base — even fewer than LoRA.

**Quality:** competitive on simple tasks; falls behind on complex.

**When:** extremely memory-constrained or many-variant scenarios.

---

## Prompt Tuning (`prompt_tuning.py`)

**Lester et al. 2021.** Learn soft prompt embeddings prepended in embedding space. Trainable params: `num_virtual_tokens × embedding_dim` — often <100K. Model frozen.

**Trade-offs:** extremely param-efficient; works only for large models (>10B); less flexible than LoRA.

**When:** rare in production. LoRA supersedes.

---

## Prefix Tuning (`prefix_tuning.py`)

**Li & Liang 2021.** Learned prefix inserted at every layer's KV cache (not just input). More capacity than prompt tuning. Rarely used today; LoRA covers same use cases better.

---

## Choosing a PEFT Method (`choosing_peft_method.py`)

**Default: LoRA or QLoRA**
- LoRA if base fits in FP16/BF16.
- QLoRA if not.
- Target modules: attention + MLP.
- Rank: 8-16 for most tasks.

**Quality bump: DoRA** — same setup, ~20% slower, 0.5-2% better.

**Extend context: LongLoRA** (if needed).

**Max efficiency: IA³** (only if params/memory really matter, simple task).

**Very large on constrained hardware: QLoRA** — non-negotiable for 65B+ single-GPU.

**Skip:** prompt tuning, prefix tuning — superseded.

**Ballpark for 7B fine-tune:**

| Method | Params | VRAM | Time | Quality |
|---|---|---|---|---|
| Full FT | 7B | 60 GB | 1× | Baseline |
| LoRA r=16 | ~30M | 16 GB | 0.5× | ~97% |
| QLoRA r=16 | ~30M | 8 GB | 0.7× | ~96% |
| DoRA r=16 | ~32M | 18 GB | 0.6× | ~98% |
| IA³ | ~1M | 15 GB | 0.4× | ~93% |
| Prompt Tuning | ~50K | 14 GB | 0.3× | ~85% |

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `full_finetune_vs_peft.py` | The core trade-off |
| `lora_deep_dive.py` | Math, code, intuitions |
| `lora_rank_selection.py` | Choosing r |
| `lora_target_modules.py` | Which layers to adapt |
| `qlora_deep_dive.py` | 4-bit + LoRA |
| `dora.py` | Weight-decomposed LoRA |
| `longlora.py` | Extending context |
| `adalora.py` | Adaptive rank |
| `ia3.py` | Even fewer parameters |
| `prompt_tuning.py` | Soft prompts |
| `prefix_tuning.py` | Prefix soft tokens |
| `choosing_peft_method.py` | Decision framework |

---

*Previous: [← Synthetic Data](../synthetic_data_generation/README.md) · Next: [SFT →](../supervised_finetuning/README.md)*  ·  *Back to [main README](../../README.md)*

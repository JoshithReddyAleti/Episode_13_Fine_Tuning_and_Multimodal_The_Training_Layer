# 🔗 Model Merging — Combining Models Without Training

> *You have two fine-tunes. Both are pretty good. Instead of retraining a combined dataset, you can arithmetically combine the models' weights and often get something better than either. Most cost-effective quality technique of 2024-2026.*

---

## Merging Fundamentals (`merging_fundamentals.py`)

**Merging** = combine weights of models with same architecture via math operations.

**Why it works** (partially understood):
- Fine-tunes from same base occupy nearby weight-space regions.
- Averaging/combining finds good solutions in the "middle."
- Different fine-tunes have complementary capabilities; merging captures both.

**Can merge:** same-base fine-tunes; LoRA adapters from same base; full models with same architecture.

**Can't easily:** different architectures; different tokenizers; different sizes.

**Why matters:**
- **Zero training cost** to combine capabilities.
- **Iteration in minutes** instead of days.
- **Ensembling-like benefits** without inference-time ensemble cost.
- **Top of open-source leaderboards** in 2024-2026 are largely merges.

---

## Linear and SLERP (`linear_and_slerp.py`)

**Linear (Weight Averaging):**
```
θ_merged = α · θ_A + (1 - α) · θ_B
```
Works when models close in weight space. Fails when they diverge — linear goes through low-quality "valleys."

**SLERP (Spherical Linear Interpolation):**
Interpolates along shortest arc between models as points on hypersphere. Preserves parameter vector magnitudes. Often better than linear for larger interpolations.

**When:** merging two same-base fine-tunes. α = 0.3-0.7.

**Failure:** if too dissimilar (different fine-tuning objectives), SLERP produces incoherent output too.

---

## TIES Merging (`ties_merging.py`)

**TIES (TrIm, Elect Sign, and Merge) — Yadav et al. 2023.**

**Problem:** when merging N fine-tunes, different ones may push a weight in opposite directions. Naive average cancels useful updates.

**Algorithm:**
1. **Trim.** For each fine-tune, compute delta from base. Keep only top-k% largest per layer; zero rest.
2. **Elect sign.** For each parameter, look at signs of non-zero deltas across models. Pick majority (or magnitude-weighted).
3. **Merge.** Average deltas that agree with elected sign.
4. **Apply.** `θ_merged = θ_base + averaged_deltas`.

**Result:** conflict-aware merging that preserves important updates while dropping noise and canceling opposing changes.

Measurably better than linear/SLERP for merging 3+ models.

---

## DARE (`dare.py`)

**DARE (Yu et al. 2023) — a preprocessing step.**

**Insight:** most parameters in fine-tuning delta are near zero. Randomly zero out most and rescale remaining without hurting quality — makes merging more effective.

**Algorithm:**
1. For each fine-tune's delta from base:
   a. Randomly zero fraction `p` of parameters.
   b. Rescale remaining by `1/(1-p)`.
2. Apply any merging method (linear, TIES) on sparsified deltas.

**Why:** sparsification like dropout, rescaling preserves expected magnitude. Reduces interference.

**Parameter:** p typically 0.5-0.9.

**In practice:** preprocessing step often combined with TIES ("TIES-DARE" or "DARE-TIES") for best results.

---

## MergeKit (`mergekit_deep_dive.py`)

**Charles Goddard's tool** — made model merging accessible.

**Supports:** linear, SLERP, TIES, DARE, and combinations. YAML config. Progressive merging (A+B then merge result with C). Slice-based (different methods for different layers). Handles safetensors, loading, saving.

**Example config:**
```yaml
models:
  - model: teknium/OpenHermes-2.5-Mistral-7B
  - model: mistralai/Mistral-7B-Instruct-v0.2
merge_method: ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  weight: 0.5
  density: 0.5
dtype: bfloat16
```

**Adoption:** almost every top open-source Llama/Mistral fine-tune on leaderboards in 2024-2025 was MergeKit output. Standard tool.

Repo: github.com/arcee-ai/mergekit

---

## Evolutionary Merging (`evolutionary_merging.py`)

**Sakana AI 2024.** Evolutionary algorithms to search the space of merges.

**Setup:** fitness = performance on target benchmark. Genome = merge weights, methods, per-layer params. Population of configurations. Evolve.

**Result:** finds non-obvious merge configurations outperforming hand-designed.

**Cost:** hundreds of merge+eval cycles. Total compute significant but each cycle cheap.

**When:** >2 candidate models and unclear which combination best; fast automatic eval metric; budget for many small runs.

Tools: Sakana's `evolve-merge` and community forks.

---

## When Merging Beats Fine-Tuning (`when_merging_beats_finetuning.py`)

**Merging wins:**
1. Already have multiple domain fine-tunes.
2. Capability composition (A=code, B=math → good at both).
3. Rapid iteration (10 merge variants in time for 1 fine-tune).
4. Community fine-tunes (benefit from others' work).
5. Model surgery — transplant capability.

**Merging loses:**
1. Need capabilities requiring new data.
2. Precise control (fine-tuning more predictable).
3. Alignment (merging can destroy alignment).
4. Safety-critical (merged models sometimes exhibit unexpected behaviors).

**Real-world 2025 pattern:**
1. Take strong base.
2. Fine-tune 2-4 different ways (different data mixes, methods).
3. Merge with TIES-DARE.
4. Optionally: light SFT/DPO on merged.

Multiple top leaderboard positions in 2024-2025 followed exactly this.

**Caveat:** merging complements serious fine-tuning of a well-designed system — doesn't replace it. Enterprises with production systems still need proper fine-tuning pipelines.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `merging_fundamentals.py` | What it does and why |
| `linear_and_slerp.py` | Simplest methods |
| `ties_merging.py` | Conflict-aware |
| `dare.py` | Drop and rescale |
| `mergekit_deep_dive.py` | Standard tool |
| `evolutionary_merging.py` | Automated search |
| `when_merging_beats_finetuning.py` | Decision framework |

---

*Previous: [← LoRA Adapter Serving](../lora_adapter_serving/README.md) · Next: [Multimodal Foundations →](../multimodal_foundations/README.md)*  ·  *Back to [main README](../../README.md)*

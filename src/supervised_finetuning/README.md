# 🎓 Supervised Fine-Tuning — Doing SFT Right

> *SFT looks simple: take an instruction dataset, run trainer.train(), get a model. The number of subtle failures in that seven-word description is why most SFT projects ship worse-than-expected results.*

---

## SFT Fundamentals (`sft_fundamentals.py`)

SFT is standard next-token prediction (cross-entropy) applied to instruction-response pairs.

**Loss for sequence t₁, t₂, ..., tₙ:**
```
L = -Σᵢ log P(tᵢ | t₁, ..., tᵢ₋₁)
```

Modified by **loss masking** — user's input tokens excluded from loss so model learns to *produce* good outputs, not reproduce inputs.

**Training loop:**
1. Load base model.
2. Configure LoRA (attach adapter layers).
3. Iterate: tokenize with chat template → apply loss mask → forward → backward → update.
4. Evaluate on validation periodically.
5. Save checkpoint.

**Frameworks:** `trl.SFTTrainer` (most common), Axolotl (config-driven), LlamaFactory, Unsloth (memory-optimized), PyTorch Lightning (from scratch).

What "just works" hides: dataset formatting, chat template application, loss masking, LR scheduling, gradient accumulation, checkpoint management. Each has failure modes.

---

## The SFT Hyperparameters (`the_sft_hyperparameters.py`)

**In order of impact:**

**1. Learning rate.** Most impactful knob.
**2. Epochs.** 1-3 typical. More usually overfits.
**3. Batch size (effective).** = per-device × grad accum × devices. Target 32-128 for 7B-70B.
**4. Warmup.** Linear for first 3-10% of steps.
**5. LR schedule.** Cosine decay (default), linear, constant.
**6. Weight decay.** 0-0.1, often 0.01.
**7. Sequence length / packing.** Match your distribution.
**8. Gradient clipping.** Clip global norm at 1.0.
**9. Precision.** BF16 (matches modern tensor cores, better range than FP16).

**Starter config QLoRA 7B:**
```yaml
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4    # effective batch 16
warmup_ratio: 0.03
lr_scheduler_type: cosine
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
```

---

## Learning Rate — The Single Most Important Knob (`learning_rate_selection.py`)

**Ranges:**
- **Full FT:** 1e-6 to 1e-5.
- **LoRA:** 1e-5 to 1e-4.
- **QLoRA:** 1e-4 to 5e-4.
- **Prompt tuning:** 1e-3 to 1e-2.

**Higher LR when:** fewer trainable params, smaller dataset, task far from pretraining.

**Lower LR when:** more trainable params, larger dataset, task close to pretraining.

**How to find:**
1. **LR range test:** ramp LR from 1e-6 to 1e-2 over a few hundred steps. Optimal is ~10× lower than divergence point.
2. **Small sweep:** try {1e-5, 5e-5, 1e-4, 5e-4} on subset.
3. **Published values:** start from similar setups.

**Symptoms:**
- Too high: loss doesn't decrease or spikes to NaN.
- Too low: loss decreases slowly, plateaus early.
- Right: smooth decrease over first 100 steps.

**Rule:** LR selection is the #1 debugging point when SFT isn't working.

---

## Batch Size and Gradient Accumulation (`batch_size_and_gradient_accum.py`)

**Effective batch = per_device × devices × grad_accum.**

**Trade-offs:** larger = more stable but slower iteration; smaller = noisier (regularization-like).

**Per-device is memory-limited.** 7B 2K context on 24GB: per-device 2-4.

**Typical effective batches:** 7B 16-64; 13B-30B 32-128; 70B 64-256.

---

## Training Stability (`training_stability.py`)

**Common failures and diagnoses:**

**Sudden loss spike** → aberrant example. Find and fix.
**Slow-drift instability** → LR too high, clipping too loose, precision issue.
**NaN loss** → exploding gradients, FP16 overflow, division by zero. Switch to BF16, add clipping.
**Loss plateaus early** → LR too low or need more data.
**Loss decreases but eval doesn't** → dataset/eval mismatch.

**Debug tools:** log per-step gradient norms, activation statistics, LR values. Use `torch.autograd.detect_anomaly()` if numerical.

---

## Catastrophic Forgetting (`catastrophic_forgetting.py`)

Fine-tuning on narrow task can make model worse elsewhere.

**When it matters:** narrow domain SFT, long training on small data, full FT.

**Detection:**
- **Regression test suite:** held-out covering base capabilities (math, general knowledge, code, reasoning). Compare before/after.
- **Standard benchmarks** (MMLU, HellaSwag, ARC).

**Mitigation:**
1. PEFT (LoRA) — frozen base preserves better than full FT.
2. Include general data in mix (5-20%).
3. Lower LR.
4. Fewer epochs.
5. Regularization.
6. Distillation from base.

**Rule:** every fine-tuning project needs a regression suite. Every one.

---

## Packing and Padding (`packing_and_padding.py`)

Variable-length examples padded to max_seq_length waste compute.

**Padding (naive):** every example padded. Efficiency bad when length distribution wide.

**Packing:** concatenate multiple short examples into one sequence up to max_seq_length. Nearly 100% useful compute. Requires attention respect boundaries (masks or document boundaries).

**Support:** `trl.SFTTrainer` with `packing=True`, Axolotl "sample packing", custom PyTorch with block-diagonal masks.

**When to pack:** short-example instruction datasets. 2-4× throughput improvement.

**When not:** long-example datasets; multi-turn where you need clean turn boundaries.

---

## Loss Masking Strategies (`loss_masking_strategies.py`)

**Loss mask determines which tokens contribute.**

**1. Instruction masking (standard):** loss only on assistant tokens. User turns, system prompts, function definitions not in loss.

**2. Full sequence loss:** all tokens. Rarely used for instruction fine-tuning; used for CPT.

**3. Response-only:** multi-turn — only last assistant turn.

**4. All assistant turns:** multi-turn — every assistant turn contributes.

**Framework defaults vary.** Verify. Wrong loss masking is a common silent bug.

**Verification recipe:** after tokenization, print input_ids and their loss-mask status. Manually check only assistant response tokens have loss enabled.

---

## Chat Template Alignment (`chat_template_alignment.py`)

**Every base model expects a specific chat format.** Train with one, serve with another → erratic behavior. **The #1 fine-tuning bug in the wild.**

**Llama-3.1:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>
```

**Mistral / Mixtral:** `<s>[INST] {user} [/INST] {assistant}</s>`

**ChatML (Qwen, some Yi):**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

**Rules:**
1. Use tokenizer's `apply_chat_template`.
2. During inference, apply same template.
3. Special tokens matter — wrong EOS/BOS breaks generation.

**Debugging:** model doesn't stop → EOS misconfigured. Produces junk after turn → template mismatch. Doesn't follow instructions → possibly serving with different template.

**Verify:** tokenize a full multi-turn, decode back to text, verify matches expected format character-by-character.

---

## SFT Common Failures (`sft_common_failures.py`)

The real ways SFT goes wrong:

1. Chat template mismatch — #1 by frequency.
2. Wrong loss masking.
3. Data leakage.
4. LR too high (diverges or plateaus bad).
5. LR too low (stable but no learning).
6. Catastrophic forgetting.
7. Overfitting to noisy examples.
8. Format overfitting — model learns dataset quirks.
9. Length overfitting.
10. Reward hacking during eval.

**Launch checklist:**
- [ ] Loss decreases smoothly.
- [ ] Val loss reasonable and stable.
- [ ] Task metrics improve on held-out.
- [ ] Regression suite passes.
- [ ] Chat template verified end-to-end.
- [ ] Loss mask verified via manual inspection.
- [ ] Golden set evaluated (only after final model chosen).
- [ ] Human eval on samples.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `sft_fundamentals.py` | Loss, training loop, frameworks |
| `the_sft_hyperparameters.py` | LR, batch, warmup, schedule |
| `learning_rate_selection.py` | The single most important knob |
| `batch_size_and_gradient_accum.py` | Effective batch tuning |
| `training_stability.py` | Debugging spikes and NaN |
| `catastrophic_forgetting.py` | Losing base capabilities |
| `packing_and_padding.py` | Efficient training |
| `loss_masking_strategies.py` | Instruction masking done right |
| `chat_template_alignment.py` | The #1 bug |
| `sft_common_failures.py` | The launch checklist |

---

*Previous: [← PEFT](../parameter_efficient_finetuning/README.md) · Next: [Preference Optimization →](../preference_optimization/README.md)*  ·  *Back to [main README](../../README.md)*

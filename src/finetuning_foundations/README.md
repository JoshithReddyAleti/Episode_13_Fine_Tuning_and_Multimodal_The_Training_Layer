# 🧭 Fine-Tuning Foundations — Why This Discipline Exists

> *Most fine-tuning projects fail. The failure isn't technical — the training runs to completion. The failure is that the project shouldn't have been a fine-tune in the first place. This section is the decision layer.*

---

## Why Fine-Tune At All (`why_fine_tune_at_all.py`)

Fine-tuning changes the model's weights. It's a *last-resort* technique in modern AI engineering because every alternative is faster, cheaper, and easier to change.

**The legitimate reasons to fine-tune:**

1. **Style/tone/format that prompting can't reliably produce.** You need every output in a specific corporate voice, structure, or domain vernacular, and even few-shot examples aren't enough.
2. **Domain vocabulary the model doesn't know.** Medical, legal, financial, or company-internal terminology where the base model consistently misinterprets terms.
3. **Structured extraction tasks with high volume.** JSON-schema outputs at high throughput where every constrained-decoding token costs money — fine-tuning removes the need for the constraint machinery.
4. **Latency where a smaller fine-tuned model beats a bigger prompted one.** A well-fine-tuned 7B often beats a prompted 70B on a narrow task at 10× lower cost.
5. **Behaviors that are hard to describe.** Alignment to preferences that are easier to demonstrate than instruct (preference optimization territory).
6. **Refusing to answer things.** A fine-tuned model can be more reliably guardrail-adherent than a system-prompt-only guarded model.
7. **Adapting to a new modality.** Vision-language fine-tuning of a text model, or extending context length via continued pretraining.

**The illegitimate reasons — where teams burn months and then admit prompting would have worked:**

- "The model doesn't know our product." (RAG fixes this.)
- "The model is too generic." (Better prompting + few-shot fixes most of this.)
- "We want better accuracy." (Vague. Measure the gap against a strong prompt baseline first.)
- "Everyone else fine-tunes." (Not a reason.)
- "We fine-tuned once and it was cool." (Sunk cost.)
- "The client asked for a custom model." (Educate them on TCO before agreeing.)

**Rule:** if you can't articulate what fine-tuning fixes that better prompting + retrieval couldn't, don't fine-tune.

---

## Prompt vs RAG vs Fine-Tune — The Decision Framework (`prompt_vs_rag_vs_finetune.py`)

The three levers, ranked by cost/complexity:

| Approach | What it changes | Cost | Change velocity | When to use |
|---|---|---|---|---|
| **Prompting** | Nothing about the model | Minutes | Instant | First choice for behavior shaping |
| **RAG** | The model's context, not weights | Days to build | Hourly-daily updates | When missing knowledge is the problem |
| **Fine-tuning** | Model weights | Weeks | Monthly release cycles | When the *behavior* itself is the problem |

**The Karpathy heuristic** (widely cited): *prompting for behavior control, RAG for knowledge, fine-tuning for style/format/skill you can demonstrate but not describe.*

**The decision flow:**
1. Can you describe what you want in a system prompt? Try that. Measure quality on a golden set.
2. Not enough? Add few-shot examples.
3. Still not enough because *the model doesn't know facts you need*? Add RAG.
4. Still not enough because *the model consistently misses the style, format, or subtle skill*? Now fine-tuning is on the table.
5. Even then: try prompted with strongest available base model first. If prompted GPT-4-class beats your target, ask "do we self-host a smaller fine-tuned model to cut cost?"

The wrong sequence — jumping straight to fine-tuning — is where most fine-tune projects that get killed come from.

---

## The Fine-Tuning Lifecycle (`the_finetuning_lifecycle.py`)

Fine-tuning isn't `trainer.train()`. It's a lifecycle:

```
1. Problem framing         → what specifically is fine-tuning fixing?
2. Baseline measurement    → prompt/RAG baseline on golden set
3. Success criteria        → what number must move? by how much?
4. Dataset construction    → biggest lever, biggest time sink
5. Method selection        → SFT? DPO? Continued pretrain? PEFT variant?
6. Infrastructure setup    → single GPU? multi-GPU? cloud?
7. Training runs           → often 5-20 iterations before shipping
8. Evaluation              → task metrics + regression + human review
9. Serving decisions       → merge into base? serve as LoRA?
10. Monitoring in prod     → does the model still perform as expected?
11. Retraining pipeline    → this is not one-shot
```

**Time breakdown (typical):** Dataset 50-70%, evaluation setup 10-15%, training runs 10-15%, infrastructure 5-10%, serving 5-10%.

**People who spend 90% on training and 10% on the rest ship the worst fine-tunes.**

---

## Economics of Fine-Tuning (`economics_of_finetuning.py`)

Real costs. Numbers are indicative for mid-2026 pricing.

### Training compute
- **QLoRA on 7B, small dataset (10K examples):** 4-8 hours on 1×A100 → ~$15-40.
- **QLoRA on 70B, 40K examples:** ~24 hours on 4×A100 → ~$150-400.
- **Full fine-tune of 7B, 40K examples:** ~12 hours on 8×A100 → ~$400-800.
- **Continued pretraining, 7B, 5B tokens:** ~40 hours on 8×A100 → ~$1200-2500.

### Human-in-the-loop costs
- **Instruction dataset labeling:** ~$1-3 per example. 10K examples = $10-30K.
- **Preference pair labeling:** ~$0.50-2 per pair. 5K pairs = $2.5-10K.
- **Golden evaluation set:** small (200-500) but high quality (~$5-15 each) = $1-8K.

### Iteration costs
Rarely one training run. **5-15 iterations typical** before shipping.

### Total realistic TCO
- **Smallish domain fine-tune (7B QLoRA, first launch):** $30-80K including data.
- **Larger (70B fine-tune with DPO):** $150-400K.

**Rule:** if the equivalent 12-month API cost is less than the fine-tune's TCO, don't fine-tune — pay the API and revisit annually.

---

## Common Fine-Tuning Myths (`common_finetuning_myths.py`)

**Myth 1: "More data is always better."** No. Better data is better. 5,000 clean, task-representative examples usually beat 50,000 noisy ones.

**Myth 2: "Fine-tuning teaches the model new facts."** Rarely. Fine-tuning teaches *behaviors, styles, and patterns*. For facts, use RAG.

**Myth 3: "Bigger models fine-tune better."** For a given task, a well-tuned 7B often outperforms a poorly-tuned 70B.

**Myth 4: "PEFT is worse than full fine-tuning."** For most task-specific fine-tunes, LoRA/QLoRA achieves 95-99% of full FT quality at 1-10% of the cost.

**Myth 5: "Fine-tuning removes the need for guardrails."** Absolutely not. Fine-tuned models still need safety layers.

**Myth 6: "Fine-tune once and you're done."** The base model improves. Your product evolves. Distribution shifts. **Retraining cadence is a first-class product concern.**

**Myth 7: "The training loss going down means it's working."** Training loss going down means the model is memorizing your training set. Evaluate on held-out task metrics, not loss.

**Myth 8: "Alignment training (DPO/PPO) makes the model 'smart'."** It makes the model prefer certain outputs. It won't fix knowledge gaps or reasoning errors.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `why_fine_tune_at_all.py` | Real reasons vs vibes |
| `prompt_vs_rag_vs_finetune.py` | The decision framework |
| `the_finetuning_lifecycle.py` | 11-step end-to-end |
| `economics_of_finetuning.py` | Real costs across phases |
| `common_finetuning_myths.py` | Where teams get it wrong |

---

*Next: [Dataset Construction →](../dataset_construction/README.md)*  ·  *Back to [main README](../../README.md)*

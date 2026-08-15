# 📊 Dataset Construction — The Unsexy 80% Of Fine-Tuning Success

> *Every failed fine-tune I've seen was a dataset problem. Every successful one was a dataset that had been agonized over. Compute is cheap; data is where the work lives.*

---

## Dataset Design Principles (`dataset_design_principles.py`)

A training dataset is not "examples of the task" — it's a *statistical distribution* the model will learn to reproduce.

**The three pillars:**

**1. Distribution — matches your production traffic.** If 60% of production is short factual, 40% long conversational, your dataset should approximate that ratio. Sample your dataset's characteristics (length distribution, topic mix, complexity) and compare to production. If they diverge, the model will generalize badly.

**2. Coverage — hits the edges.** Rare-but-important cases need explicit representation. Adversarial cases (typos, hostile phrasing, ambiguous instructions) need to be there. Negative examples — where correct is "I don't know" or "I can't help" — need explicit coverage.

**3. Quality — clean and consistent.** Consistency of format, persona, voice. A single wrong label poisons more than one training step.

The tension: distribution needs volume; coverage needs specific hard cases; quality needs manual review. Balancing these is the actual craft.

---

## Task Specification (`task_specification.py`)

Before writing any examples, write the **task spec** — the single most under-invested step.

**A task spec includes:**
- The task in one paragraph, precise.
- Input format: exact structure, ranges, edge cases.
- Output format: JSON schema, character limits, allowed values.
- Behavior for edge cases: malformed input, ambiguous queries, offensive content, out-of-scope requests.
- Style guidelines: tone, length, voice, forbidden phrases.
- Side-by-side "good" and "bad" examples.
- Ambiguity resolution rules.

**Bootstrap sequence:**
1. Write the task spec.
2. Have 3 people annotate the same 20 examples using only the spec.
3. Measure agreement.
4. Wherever they disagree, the spec is under-specified. Revise.
5. Repeat until agreement is high.
6. Only *then* start large-scale labeling.

Skipping this step means every downstream metric is noisy.

---

## Data Collection Strategies (`data_collection_strategies.py`)

Sources, roughly in order of quality-per-dollar:

1. **Production traffic (with consent + PII scrubbing).** Real users, real inputs. Best data.
2. **Historical customer data.** High-value, needs anonymization.
3. **Public datasets.** Alpaca, Dolly, OpenAssistant, ShareGPT, UltraFeedback, HH-RLHF. Good for general capabilities. Beware contamination.
4. **Vendor-labeled.** Scale AI, Surge AI, Toloka, Labelbox. Expensive but consistent.
5. **Crowdsourced.** Mechanical Turk, Prolific. Cheaper but variable quality.
6. **Synthetic (from a larger model).** Cheap, unlimited scale, quality depends on filter pipeline.
7. **Employee/expert-generated.** Highest quality per example, most expensive. Reserved for golden eval and seed.

**Real-world composition** in production fine-tunes: seed set 100-500 expert-labeled, bulk 5-50K synthetic or vendor-labeled, preference data 2-20K pairs, golden eval 200-500 (kept separate).

---

## Labeling Workflows (`labeling_workflows.py`)

**Three patterns, most-to-least common:**

1. **Human-generated** — labelers write inputs and outputs from scratch. Highest quality, slowest. Used for seed set and preference pairs.
2. **LLM-generated + human review** — capable LLM generates candidates; humans accept/edit/reject. **10-30× throughput** vs pure human. The dominant pattern in 2026.
3. **LLM-only with automated filters** — no human touch. Fastest, cheapest. Quality risk is real.

**Tools:** Label Studio, Argilla, Prolific, Scale Rapid, Weights & Biases annotation queues.

**Inter-annotator agreement (IAA):** Cohen's kappa or Krippendorff's alpha. Below κ=0.6 → spec problem. Above κ=0.8 → strong.

If IAA is bad, your model's ceiling is bad. Fix this before scaling.

---

## Data Quality Metrics (`data_quality_metrics.py`)

Measurable properties of a dataset:

- **Length distribution** — inputs and outputs.
- **Duplication rate** — MinHash near-duplicates. Aim <5%.
- **Format consistency** — do outputs follow the schema?
- **Toxicity / safety** — Perspective API, Detoxify, or classifier.
- **PII presence** — Presidio, custom regex, or LLM classifier.
- **Perplexity from base model** — outliers may be noise.
- **Semantic diversity** — embed all inputs, cluster, measure spread.
- **Label balance** — for classification-like tasks.

Run these on every dataset version. Track them like performance metrics.

---

## Data Cleaning at Scale (`data_cleaning_at_scale.py`)

Standard pipeline:

1. Deduplicate (exact, near-match via MinHash, semantic).
2. Filter garbage (empty inputs, HTML fragments, encoding errors).
3. Normalize (line endings, whitespace, punctuation, encoding).
4. PII scrub (emails, phones, names via NER, credit cards).
5. Toxicity filter.
6. Length filter (bounds).
7. Language filter (fasttext lid.176, cld3).
8. Contamination check.
9. Format validation.
10. Human spot-check (random 100 as final gate).

Each stage removes 5-30%. Real-world attrition from raw scraped → training-ready is often **50-80% loss**.

Tools: `datatrove` (Meta), `dolma` (AllenAI), `datasets` library with custom filters.

---

## Data Augmentation (`data_augmentation.py`)

**When:** low volume for a category, adversarial robustness needed, domain gap.

**How:** back-translation (EN → FR → EN paraphrase), LLM paraphrasing, template expansion, noise injection (typos), format variation.

**Warning:** augmentation is *not* a substitute for real data volume. Augmenting 100 examples 10× gives you a 100-flavor dataset, not a 1,000-flavor one.

---

## Instruction Dataset Formats (`instruction_dataset_formats.py`)

**Alpaca format** (single-turn), **ShareGPT format** (multi-turn), **OpenAI messages format** (now dominant), **ChatML** (raw text form once templates applied).

**Practical guidance:**
- Store in OpenAI messages format — easy to convert to any framework.
- Apply the model-specific chat template at training time.
- **Getting the chat template wrong is the #1 fine-tuning bug.**

---

## Preference Dataset Creation (`preference_dataset_creation.py`)

For DPO/RLHF, you need pairwise preferences: `{"prompt": ..., "chosen": ..., "rejected": ...}`.

**Sources:**
- Human-labeled pairs.
- LLM-labeled pairs (fast, biased toward judge's preferences).
- Rule-based (constrained tasks with automatic verifiability: code passing tests, math correct).
- Derived from existing SFT (multiple checkpoints, sampling temperatures, then judge).

**Key design:**
- 2K-20K pairs typical for domain DPO.
- **Balance for surface features.** If chosen answers are systematically longer, DPO teaches "prefer longer" not "prefer better."

---

## Dataset Versioning (`dataset_versioning.py`)

Every training run must reference an exact dataset version.

**Options:** DVC, HuggingFace Datasets Hub, MLflow, S3 + hash, W&B Artifacts.

**Practical:** immutable dataset artifacts. Never overwrite; always new version. Each training run logs the dataset version SHA. Dataset changes get PR review — **dataset is code**.

---

## Train/Val/Test Splits (`train_val_test_splits.py`)

**Correct splitting is trickier than it sounds:**

- **Not random for grouped data.** Multiple examples from same source → split by source, not example.
- **Time-based splits** for time-sensitive tasks.
- **Stratified splits** by category, difficulty, source.
- **Held-out golden set** — never see it during any iteration.

**Typical proportions:** train 80-90%, val 5-10% (early stopping), test 5-10% (final eval). Plus separate small golden eval (200-500).

**Contamination discipline:** the golden eval set is built before training data collection starts, and locked in a separate bucket with restricted access.

---

## The Dataset Iteration Loop (`the_dataset_iteration_loop.py`)

Fine-tuning is iterative:

```
1. Train on current dataset.
2. Evaluate on held-out (task metrics + regression).
3. Failure analysis: WHERE does the model fail?
   - By category? Underrepresented.
   - By length? Length mismatch.
   - By difficulty? Hard cases underrepresented.
   - By format? Format inconsistency.
4. Update the dataset (add failing examples, fix labels, rebalance).
5. Retrain.
6. Go to 2.
```

**The failure analysis step is where most teams cheat** by looking at 3 examples. Rigorous analysis means categorizing 100+ failures, quantifying patterns, adding targeted data.

**Stopping criteria:** task metric hits target; marginal iterations show <1% improvement; regression suite fails.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `dataset_design_principles.py` | Distribution, coverage, quality |
| `task_specification.py` | Writing a spec before labeling |
| `data_collection_strategies.py` | Sources ranked |
| `labeling_workflows.py` | Human, LLM-assisted, automated |
| `data_quality_metrics.py` | Measurable properties |
| `data_cleaning_at_scale.py` | The 10-stage pipeline |
| `data_augmentation.py` | When and how |
| `instruction_dataset_formats.py` | Alpaca, ShareGPT, ChatML, OpenAI |
| `preference_dataset_creation.py` | For DPO/RLHF |
| `dataset_versioning.py` | Reproducibility discipline |
| `train_val_test_splits.py` | Without leakage |
| `the_dataset_iteration_loop.py` | Iterating with signal |

---

*Previous: [← Foundations](../finetuning_foundations/README.md) · Next: [Synthetic Data →](../synthetic_data_generation/README.md)*  ·  *Back to [main README](../../README.md)*

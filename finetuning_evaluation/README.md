# 📏 Fine-Tuning Evaluation — Did It Actually Work?

> *Training loss going down means you fit the data. It says nothing about whether the model is better at your task. Every metric that isn't held-out task performance is a proxy — and proxies lie.*

---

## Evaluation Beyond Loss (`evaluation_beyond_loss.py`)

**Chain of correct interpretation:**
1. Training loss decreases → model learning training data.
2. Val loss decreases → generalizes to held-out same distribution.
3. Task metrics improve on held-out → better at actual task.
4. Regression suite passes → base capabilities preserved.
5. Human eval agrees → subjective quality confirmed.
6. A/B in production → real users benefit.

**Common failure:** training loss great, val fine, task metric up... users complain. Eval was measuring wrong thing.

**Rule:** every fine-tune needs *at least four* eval layers: task benchmarks, regression suite, human/LLM-judge on real prompts, A/B in production.

---

## Benchmark Evaluation (`benchmark_evaluation.py`)

**Standard benchmarks:**
- **MMLU** — 57-subject multiple choice. Broad knowledge. Contamination-prone.
- **HellaSwag** — commonsense completion. Robust.
- **ARC-Challenge/Easy** — grade-school science.
- **TruthfulQA** — truthful answers on adversarial.
- **GSM8K** — math word problems. CoT-sensitive.
- **HumanEval/MBPP** — Python code. Pass@1.
- **MT-Bench** — multi-turn quality, LLM-judged.
- **AlpacaEval / AlpacaEval 2** — pairwise vs reference.
- **IFEval** — instruction-following.

**How to interpret:** useful for **regression testing** (base capability survived?); **poor for measuring fine-tune quality on your task**. Contamination is real.

**Tools:** `lm-evaluation-harness` (EleutherAI standard), `evalplus` (code), `open-llm-leaderboard` (HF aggregate).

---

## Task-Specific Evaluation (`task_specific_evaluation.py`)

**What your fine-tune actually needs to be good at.**

**Design principles:**
1. Represent production distribution.
2. Diverse difficulty.
3. Include known-hard cases.
4. Held-out from training.
5. Task-appropriate metrics (not loss):
   - Extraction: field-level accuracy, F1.
   - Classification: accuracy, F1, per-class.
   - Structured generation: schema compliance + content correctness.
   - Q&A: exact match, F1 span, LLM-judged.
   - Open generation: LLM-as-judge multi-criteria + human.

**Size:** 200-2000 examples. Version-controlled. Automated every training run.

---

## Regression Testing (`regression_testing.py`)

**Check that catches catastrophic forgetting.**

Your fine-tune should improve target task without destroying base capabilities.

**Suite includes:** general knowledge (MMLU), reasoning (HellaSwag, ARC), math (GSM8K), code (HumanEval), instruction following (IFEval), safety (refusal on inappropriate).

**How:** fixed subset ~100-500 per capability. Base → baseline. Fine-tuned → compare. Threshold: any capability drop >5% → stop, investigate.

**When:** every checkpoint before shipping. Automated. Alerts on regressions.

**Real value:** many fine-tunes ship with "12% task improvement" and never mention "3% MMLU drop, 8% GSM8K drop." Regression suites make it visible.

---

## LLM-as-Judge for Fine-Tunes (`llm_as_judge_for_finetunes.py`)

**When "correct answer" is subjective, LLM judges scale.**

**Patterns:**

**Pairwise:** give judge two responses. "Which better and why?" Report win rate. Watch: position bias (A vs B), length bias, self-bias.

**Scoring:** rate 1-5 on multiple criteria. Report average per criterion. Watch: score compression.

**Rubric-based:** detailed rubric. Judge follows. Most consistent.

**Mitigating biases:** randomize position; strong judge (GPT-4-class+); multi-judge; human calibration on 50-100 samples.

**Cost:** $100-1000 per eval run.

**Use for:** chat quality, style/tone, open generation, ranking. **Don't use for:** factual correctness (unless judge can verify), code (use test execution), math (symbolic verification).

---

## Human Evaluation (`human_evaluation.py`)

**Required for:** definitive quality on open generation, safety-critical, establishing benchmarks for automated eval, UX testing.

**Structure:** rubric-based scoring (same as LLM judge for comparison), multiple raters per example (3-5, average, report agreement), blinded, domain experts for domain quality.

**Cost:** $2-10/example general; $10-50+ for expert.

**Volume:** 100-500 typical.

**Vendors:** Scale AI, Surge AI, Toloka, in-house, Prolific.

**When human and LLM judge disagree:** trust human.

---

## Evaluation Dataset Design (`evaluation_dataset_design.py`)

**Good eval:**
- Held out from training. No leakage.
- Distributionally realistic.
- Diverse. Edge-heavy.
- Stable — same version across many runs.
- Documented — what each example tests.

**Composition:** 60% normal, 20% edge, 10% adversarial, 10% negative (correct = "I don't know").

**Build:** start with production samples; expert curation for gaps; never touch during iteration.

---

## Contamination Checking (`contamination_checking.py`)

**Test-set leakage kills validity.** How it happens: public datasets teacher saw during pretraining; augmentation recreating tests; annotators copying from internet; version confusion.

**Detection:**
- Exact/substring match.
- N-gram overlap (13-gram threshold).
- Embedding proximity.

**What to do:** report contamination scores; remove contaminated tests; best — private expert-curated eval teacher couldn't have seen.

**Rule:** publishing benchmark results without contamination discussion → numbers suspect.

---

## Fine-Tuning Eval Playbook (`finetuning_eval_playbook.py`)

**Before training:**
1. Build task-specific eval set (200-2000).
2. Build regression suite.
3. Freeze golden human-eval (100-500, never touched).
4. Define pass/fail thresholds.

**During training (per checkpoint):**
1. Task-specific eval.
2. Regression suite.
3. Sample 50 outputs for manual review.
4. Compare to previous.
5. Investigate regressions before continuing.

**Before shipping:**
1. Full task-specific.
2. Full regression.
3. LLM-judge pairwise vs baseline (200-500 pairs).
4. Human eval on golden.
5. Cost/latency in serving conditions.

**After deployment:**
1. A/B in production.
2. Monitor task-specific metrics from live traffic.
3. Sample outputs for ongoing review.
4. Alerts on regression.

**Non-negotiables:** never touch golden set during iteration; report *all* metrics not just favorable; compare to strong baselines; reproducibility (log dataset/model/judge versions).

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `evaluation_beyond_loss.py` | Loss ≠ quality |
| `benchmark_evaluation.py` | Standard benchmarks |
| `task_specific_evaluation.py` | Your task, your metric |
| `regression_testing.py` | Catching catastrophic forgetting |
| `llm_as_judge_for_finetunes.py` | Pairwise, scoring, rubric |
| `human_evaluation.py` | Definitive signal |
| `evaluation_dataset_design.py` | Held out properly |
| `contamination_checking.py` | Test-set leakage |
| `finetuning_eval_playbook.py` | Full workflow |

---

*Previous: [← Training Infrastructure](../training_infrastructure/README.md) · Next: [LoRA Adapter Serving →](../lora_adapter_serving/README.md)*  ·  *Back to [main README](../../README.md)*

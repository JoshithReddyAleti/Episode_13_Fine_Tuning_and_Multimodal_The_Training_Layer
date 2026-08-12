# Fine-Tuning Evaluation Guide

Loss alone doesn't tell you if the model is better. Multi-layer evaluation:

## Layer 1: Training loss and val loss
Sanity check. Loss decreasing smoothly, val loss not diverging.

## Layer 2: Task-specific benchmarks
Your task, your metric. 200-2000 examples held out. Automated. Ran every training run.

## Layer 3: Regression suite
Fixed subset of MMLU/HellaSwag/ARC/HumanEval/GSM8K/IFEval to detect catastrophic forgetting.

## Layer 4: LLM-as-judge
Pairwise or scoring vs baseline. Use strong judge (GPT-4-class+). Randomize position, multi-judge to reduce bias.

## Layer 5: Human evaluation
Domain experts. Rubric-based. 3-5 raters per example. The definitive quality signal.

## Layer 6: A/B in production
Real users. Real cost/latency. The ultimate test.

## Non-negotiables

- Golden set untouched during iteration.
- Report all metrics, not just favorable.
- Compare to strong baselines.
- Reproducibility: log dataset/model/judge versions.
- Contamination check before publishing benchmark results.

## Common failures caught only by multi-layer eval

- Task metric up, regression suite fails → catastrophic forgetting.
- Training loss down, task metric flat → data mismatch.
- Task benchmark up, human eval flat → benchmark not measuring what matters.
- Human eval great, A/B disappointing → prompt/interface differences.

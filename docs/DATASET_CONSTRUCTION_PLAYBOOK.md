# Dataset Construction Playbook

## Phases

**Phase 1: Task spec.** Write a precise spec (input/output format, edge cases, style). 3-annotator sanity test. Iterate until agreement is high.

**Phase 2: Seed set.** 100-500 expert-authored examples covering the distribution.

**Phase 3: Bulk generation.**
- LLM-generated with human review (dominant).
- Vendor-labeled (Scale, Surge).
- Synthetic (Self-Instruct, Evol-Instruct, distillation).

**Phase 4: Quality pipeline.**
- Dedup (exact, MinHash, semantic).
- Filter (format, length, toxicity, PII, perplexity, coherence).
- Contamination check.
- Human spot-check.

**Phase 5: Preference data** (for DPO).
- 2-20K pairs.
- UltraFeedback pattern common.

**Phase 6: Splits.**
- Train/val/test with source grouping.
- Held-out golden eval built before anything else.

**Phase 7: Versioning.**
- Immutable artifacts.
- SHA logged with every training run.

## Metrics per dataset version

- Length distribution (input/output).
- Duplication rate (target <5% near-duplicates).
- Format conformance rate.
- Toxicity/PII flag rate.
- Semantic diversity.
- Contamination score vs golden.

## Iteration signal

- Fine-tune current dataset.
- Failure analysis: category, length, difficulty, format.
- Targeted data added to fix diagnosed failures.
- Retrain, re-evaluate.
- Stop when target metric hit or marginal improvement <1%.

# Alignment Method Comparison

| Method | Data | Models in mem | Complexity | Notes |
|---|---|---|---|---|
| PPO | Pairs + trained RM | 4 (policy, value, ref, RM) | High | Original RLHF |
| DPO | Pairs (offline) | 2 (policy, ref) | Low | Modern default |
| IPO | Pairs | 2 | Low | Fixes DPO overfitting |
| KTO | Binary labels | 2 | Low | Point-wise data |
| ORPO | Pairs during SFT | 1 (no ref) | Low | Single-stage |
| SimPO | Pairs | 1 (no ref) | Low | Memory-efficient |

## Decision tree

- **Have preference pairs, want simple + reliable** → DPO.
- **DPO overfits** → IPO.
- **Only thumbs up/down data** → KTO.
- **Doing SFT from scratch, have pairs** → ORPO (single-stage).
- **Memory-constrained** → SimPO.
- **Frontier quality with mature RM** → PPO.

## Evaluation

- AlpacaEval / MT-Bench.
- Task-specific golden set.
- Regression suite (did base capabilities survive?).
- Human eval on real prompts.
- Length distribution (aligned models often verbose).
- Refusal behavior (some methods over-refuse).
- A/B in production.

## Failure modes

- Sycophancy.
- Verbosity.
- Over-refusal.
- Style flattening.
- Reward hacking (gaming implicit reward).

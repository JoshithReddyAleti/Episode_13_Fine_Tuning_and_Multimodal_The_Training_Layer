# When To Fine-Tune — The Decision Guide

## Try in order

1. **Prompting** with a strong base. Measure on golden set.
2. **Few-shot examples** in the prompt.
3. **RAG** if the gap is knowledge.
4. **Fine-tuning** only if 1-3 don't reach the bar.

## When fine-tuning is right

- Style/tone/format prompting can't reliably produce.
- Domain vocabulary the base doesn't know.
- Structured extraction at high volume.
- Latency: smaller fine-tuned model beats bigger prompted one.
- Behaviors hard to describe.
- Guardrail-adherent refusals.

## When fine-tuning is wrong

- Model needs updated facts → RAG.
- Task can be solved with good prompt + best base → keep prompting.
- Budget can't cover $30-80K+ TCO for the first launch.
- No team to maintain retraining + eval infrastructure.
- Task is safety-critical without a mature eval discipline.

## The TCO breakdown

- Data collection/labeling: 40-60% of budget.
- Training compute: 10-20%.
- Evaluation infrastructure: 10-15%.
- Serving/deployment: 10-15%.
- Ongoing retraining: monthly-quarterly cadence.

## Rule of thumb

If the equivalent 12-month API cost is less than the fine-tune TCO, pay the API and revisit.

# Synthetic Data Pipeline

Teacher-model generation → 10-stage filter → dedup → contamination check → training data.

## Steps

1. Curate seed set (200-500 human-authored examples).
2. Prompt teacher (GPT-4/Claude) with Evol-Instruct-style transformations.
3. Apply 10-stage quality filter.
4. MinHash + semantic deduplication.
5. Contamination check against golden eval set.
6. Human spot-check on random 100.
7. Version and freeze; use for fine-tuning.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

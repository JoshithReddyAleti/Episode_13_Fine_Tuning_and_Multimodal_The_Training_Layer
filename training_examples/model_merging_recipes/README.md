# Model Merging Recipes with MergeKit

Compare linear, SLERP, TIES, DARE on the same set of fine-tunes.

## Steps

1. Select 2-4 fine-tunes of the same base.
2. Write MergeKit YAML configs for each method.
3. Run each merge (linear, SLERP, TIES, TIES-DARE).
4. Evaluate all merged models on same benchmark.
5. Report which method won.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

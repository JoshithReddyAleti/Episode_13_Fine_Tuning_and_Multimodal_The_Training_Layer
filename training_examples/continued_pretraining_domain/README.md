# Continued Pretraining on Domain Corpus

Adapt a general model (Llama-3.1-8B) to a specialized domain (medical/legal/finance).

## Steps

1. Curate domain corpus (5-30B tokens).
2. Preprocess: dedup, quality filter, PII scrub, tokenize.
3. Mix 70-95% domain + 5-30% general data.
4. CPT with low LR (1e-5 to 1e-4), 1 epoch on large corpus.
5. Evaluate: domain benchmarks + regression suite.
6. Optional: SFT on domain instructions after CPT.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

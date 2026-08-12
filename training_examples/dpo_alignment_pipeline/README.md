# SFT → DPO Alignment Pipeline

Two-stage recipe: supervised fine-tune first, then DPO on preference pairs.

## Steps

1. Prepare SFT dataset (instructions + responses).
2. Run SFT with LoRA to establish base capability.
3. Generate preference pairs (chosen/rejected).
4. Run DPO with trl.DPOTrainer on preferences.
5. Evaluate alignment (AlpacaEval + regression).

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

# QLoRA Fine-Tune of Llama-3.1-8B

End-to-end recipe: dataset → 4-bit + LoRA training → evaluation → merge → serve.

## Steps

1. Prepare dataset in OpenAI messages format.
2. Configure QLoRA (NF4, LoRA r=16, target attention + MLP).
3. Train with Axolotl or trl.SFTTrainer.
4. Evaluate on held-out task benchmark + regression suite.
5. Optional: merge adapter into base for standalone serving.
6. Serve via vLLM (base + adapter OR merged).

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

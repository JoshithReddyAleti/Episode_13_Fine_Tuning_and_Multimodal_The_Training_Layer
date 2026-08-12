# VLM Fine-Tuning Recipe (Qwen2.5-VL)

LoRA fine-tune of Qwen2.5-VL-7B on a domain visual task (e.g., invoice extraction).

## Steps

1. Prepare data: (image, instruction, response) triples in LLaVA format.
2. Freeze vision encoder; LoRA on LLM part (rank 32-64).
3. Train with Axolotl multimodal extension or custom HF loop.
4. Evaluate: domain VQA benchmark + hallucination check.
5. Serve via vLLM (VLM support).

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

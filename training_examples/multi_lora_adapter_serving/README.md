# Multi-LoRA Adapter Serving with vLLM

Serve N customer adapters over one Llama-3.1-70B base.

## Steps

1. Train N per-tenant LoRA adapters (same rank, same target modules).
2. Register adapters in the registry (S3 + metadata DB).
3. Start vLLM: --enable-lora --max-loras 16 --max-lora-rank 32.
4. Route requests: X-Tenant-Id → adapter name in request payload.
5. Monitor: adapter cache hit rate, per-tenant latency/cost.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

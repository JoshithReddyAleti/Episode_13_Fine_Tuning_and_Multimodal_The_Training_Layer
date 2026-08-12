# Enterprise Document Intelligence Pipeline

PDF → layout extraction → table extraction → ColPali retrieval → VLM QA with citations.

## Steps

1. Ingest PDFs; branch: text-layer + OCR + page rasterization.
2. Layout analysis (regions, reading order).
3. Table extraction (TATR + validation).
4. ColPali indexing for visual retrieval.
5. Text embedding indexing in parallel.
6. Query: hybrid retrieval + VLM answer with page-level citations.
7. Human-in-the-loop review UI for low-confidence answers.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

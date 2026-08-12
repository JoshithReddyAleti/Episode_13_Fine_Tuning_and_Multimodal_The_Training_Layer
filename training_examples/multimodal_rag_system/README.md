# Multimodal RAG System

Text + image + audio unified retrieval + VLM answer generation.

## Steps

1. Ingest documents, images, audio files.
2. Text: chunk + embed (BGE or similar).
3. Images: embed with SigLIP.
4. Audio: transcribe (Whisper) + embed transcript.
5. Unified vector DB with modality metadata.
6. Query: fanned to all modalities, rank fusion.
7. VLM generation grounded on retrieved multimodal content.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

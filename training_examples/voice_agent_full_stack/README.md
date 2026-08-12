# Voice Agent Full Stack (STT + LLM + TTS)

Sub-800ms turn latency voice agent with interruption handling.

## Steps

1. Telephony (SIP/WebRTC) input → VAD (Silero).
2. Streaming ASR (faster-whisper).
3. Turn-end detection.
4. LLM agent (with tools).
5. Streaming TTS (XTTS or ElevenLabs).
6. Interruption handling: VAD during TTS → halt output, buffer partial context.
7. Call recording, cost attribution, escalation path.

## Files

- `README.md` — this file.
- `config.yaml` — reference training config.
- `run.sh` — orchestrator (illustrative).

## Related

See the section READMEs for concept coverage:
- Fine-tuning path: `src/parameter_efficient_finetuning/`, `src/supervised_finetuning/`, `src/finetuning_evaluation/`.
- Multimodal path: `src/vision_language_models/`, `src/document_intelligence/`, `src/audio_and_speech/`.

Back to [main README](../../README.md).

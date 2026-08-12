# Concept Reference

Fast index. Full coverage in section READMEs.

## Fine-tuning

- **SFT** — supervised fine-tuning on (input, output) pairs.
- **DPO** — direct preference optimization from (chosen, rejected) pairs, no RM.
- **PPO** — RL with reward model.
- **CPT** — continued pretraining on domain corpus.
- **LoRA** — low-rank adapter matrices trained while base frozen.
- **QLoRA** — LoRA + 4-bit base weights (NF4).
- **DoRA** — LoRA with magnitude/direction decomposition.
- **Chat template** — model-specific format for conversations.
- **Loss masking** — excluding user tokens from loss.
- **Catastrophic forgetting** — losing base capabilities during fine-tuning.
- **Merging** — combining multiple fine-tunes arithmetically.
- **TIES/DARE** — conflict-aware merging methods.
- **Adapter registry** — source of truth for LoRA adapters.
- **Multi-LoRA serving** — many adapters, one base, batched.

## Multimodal

- **VLM** — vision-language model (image+text in, text out).
- **CLIP/SigLIP** — joint image-text embeddings.
- **Q-Former** — compression of image features for LLM.
- **ColPali** — visual retrieval from documents.
- **Whisper** — speech recognition.
- **VAD** — voice activity detection.
- **Diarization** — who said what in multi-speaker audio.
- **Streaming ASR/TTS** — real-time transcription/synthesis.
- **Voice agent** — STT + LLM + TTS pipeline.
- **Multimodal RAG** — retrieval + generation across modalities.
- **ControlNet, IP-Adapter** — controlled image generation.
- **NSFW/CSAM filtering** — safety pipelines for media.

Each concept has a code stub, section README, and often a doc.

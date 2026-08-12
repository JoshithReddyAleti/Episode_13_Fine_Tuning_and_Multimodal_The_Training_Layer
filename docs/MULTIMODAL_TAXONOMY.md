# Multimodal Taxonomy

## By modality

- **Text** — the baseline.
- **Vision** — images, documents (as images).
- **Audio** — speech, music, sound effects.
- **Video** — frames + audio + time.
- **Structured** — tables, code, JSON.

## By capability

- **Understanding-only** — VLM takes image+text → text.
- **Generation-only** — image gen, TTS, video gen.
- **Bidirectional** — GPT-4o, Gemini native multimodal (understand + generate).

## By architecture

- **LLaVA-style** — vision encoder + projection + LLM.
- **Q-Former** (BLIP-2) — compression via learnable queries.
- **Cross-attention** (Flamingo) — image features into LLM layers.
- **Native multimodal** — pretrained on interleaved modalities from scratch.
- **Discrete tokenization** — everything is tokens (VQ-VAE-style).

## By use case

- Document intelligence.
- Voice agents.
- Video analysis / search.
- Image search / VQA.
- Content generation.
- Multimodal RAG.
- Accessibility.

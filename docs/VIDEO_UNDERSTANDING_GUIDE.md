# Video Understanding Guide

## Video representation

- Video = frames + audio (usually).
- Frame extraction: 1-4 fps typical for understanding (full 30 fps rarely necessary).
- Each frame → VLM patch tokens.
- Temporal representation via pooling, temporal transformers, or time embeddings.

## Cost math

- 1 min @ 1 fps @ 576 tokens/frame = 34,560 tokens.
- 10 min = 345,600 tokens.

**Compression is essential** for anything beyond short clips.

## Long-video strategies

- **Hierarchical summarization** — chunk-summarize-aggregate.
- **RAG over video chunks** — retrieve relevant, VLM answers.
- **Long-context native** — Gemini handles hours (expensive).
- **Specialized long-video models** — LongVA, LongVILA (research).

## Video search / retrieval

- Extract keyframes.
- Embed frames (CLIP/SigLIP) + transcribe audio and embed transcript.
- Unified index with timestamp metadata.
- Query fanned to visual + text, results fused.

## Production considerations

- Storage: media files are large.
- Async processing pipeline.
- Cache embeddings (permanent).
- Tier by value.
- Batch off-peak.

## Failure modes

- Object permanence issues in generation.
- Temporal reasoning is hard.
- Frame subsampling can hide critical events.
- Long videos hit context and memory limits.

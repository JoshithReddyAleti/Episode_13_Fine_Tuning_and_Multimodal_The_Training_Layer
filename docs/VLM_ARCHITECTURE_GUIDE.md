# VLM Architecture Guide

## The 3-part anatomy

1. **Vision encoder** — CLIP-style ViT produces per-patch features from image.
2. **Projection layer** — small MLP or Q-Former maps vision features to LLM's embedding space.
3. **LLM** — receives image tokens + text tokens as one sequence.

## Training stages

- **Stage 1: Alignment.** Freeze vision encoder and LLM. Train only the projection on image-caption pairs.
- **Stage 2: Instruction tuning.** Unfreeze LLM (or attach LoRA). Train on visual instructions.

## Image tokenization

- Preprocess: resize (336² / 448² / 1024²) or dynamic resolution.
- Patch: divide into 14×14 or 16×16 patches.
- Embed: linear projection per patch.
- ViT processes: contextualized patch features.
- Project: to LLM embedding dim.
- Insert: special tokens delimit position in sequence.

## Cost implications

- 336² with 14² patches → 576 tokens/image.
- 1024² → 5000+ tokens/image.
- **High-res is expensive.** Every task should decide the trade-off.

## Modern VLMs (2026)

- Commercial: GPT-4o, Claude, Gemini.
- Open: Qwen2.5-VL, Llama-3.2 Vision, Pixtral, InternVL, MiniCPM-V, Molmo.

## Fine-tuning VLMs

- LoRA on LLM part; sometimes projection.
- Vision encoder often frozen.
- Rank 32-64 (higher than text-only).
- LR 1e-4 to 5e-5, 1-3 epochs.

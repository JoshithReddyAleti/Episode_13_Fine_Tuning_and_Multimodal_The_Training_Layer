# Interview Prep

## Fine-tuning questions

**Q: When would you fine-tune vs use RAG vs improve prompting?**
Prompting first (fastest to change). RAG for knowledge gaps. Fine-tuning for style/format/skill you can demonstrate but not describe. TCO matters: if 12-month API cost < fine-tune TCO, don't fine-tune.

**Q: Explain LoRA math.**
Instead of updating W (d×k), learn ΔW = B·A where B (d×r) and A (r×k), with r << min(d,k). Effective weight W + α·B·A/r. B initialized to 0 so ΔW=0 at start. Merges cleanly back into base. Training params: d·r + r·k instead of d·k.

**Q: How do you choose LoRA rank?**
Start r=8. If underfit (both losses high) → increase. If overfit (train low, val high) → more data or lower r. Diminishing returns around r=32-64. Simple style: 4-8. Instruction: 8-16. Complex task: 16-32.

**Q: Why does QLoRA work?**
LoRA doesn't update base weights, so base only needs to be *readable*. 4-bit NF4 quantization is readable enough while saving 4× memory. Combined with double quantization and paged optimizer → 65B fits on one 48GB GPU.

**Q: Difference between DPO and PPO.**
PPO: sample from policy, score with RM, RL update. Needs RM + value + reference + policy. Complex, unstable. DPO: derives implicit reward from policy log-ratio, turns into supervised loss. No RM, no RL, just backprop.

**Q: How would you detect catastrophic forgetting?**
Regression suite of fixed subsets from MMLU, HellaSwag, ARC, HumanEval, GSM8K. Run before and after fine-tuning. Any drop >5% is a red flag. Run this on every checkpoint before shipping.

**Q: How do you serve multi-tenant fine-tunes economically?**
One base model + per-tenant LoRA adapters. vLLM multi-LoRA batches different adapters in same forward pass. 50-500× cheaper than N dedicated fine-tunes. Requires adapter registry, cache management, gateway with tenant → adapter mapping.

## Multimodal questions

**Q: How do VLMs work?**
Vision encoder (CLIP-style ViT) produces patch features. Projection MLP maps to LLM embedding space. LLM processes image tokens + text tokens as one sequence. Training in two stages: alignment (train projection) then instruction tuning (train LLM/LoRA).

**Q: What are image tokens?**
Image is divided into patches (e.g., 14×14). Each patch → linear projection → embedding. 336² image with 14² patches = 576 tokens per image. High resolution is expensive (1024² → 5000+ tokens).

**Q: How would you build a document QA system?**
Ingestion → preprocessing → (text-layer + OCR + rasterization) → layout analysis → chunk/page-level indexing (text + ColPali visual) → query rewriting → hybrid retrieval → reranking → VLM answer with page-level citations. Human-in-the-loop review for low-confidence.

**Q: What's ColPali and why does it matter?**
Document retrieval where pages are embedded as images (not OCR text). Captures layout, figures, charts that OCR misses. Multi-vector late-interaction retrieval. Growing rapidly since 2024 in document-heavy RAG.

**Q: How do you build a low-latency voice agent?**
Streaming ASR (faster-whisper) + agent LLM + streaming TTS. VAD for turn detection. Target sub-800ms end-of-speech to first audio. Optimizations: speculatively start LLM on partial ASR; pre-warm TTS with filler; interruption handling with buffered context.

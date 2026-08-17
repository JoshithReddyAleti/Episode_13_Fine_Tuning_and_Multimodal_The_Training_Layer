# 👁️ Vision-Language Models — VLMs From Fundamentals to Production

> *A VLM lets a language model see. This section is how the seeing part actually works and what to do with it.*

---

## CLIP and Its Descendants (`clip_and_its_descendants.py`)

**CLIP (Radford et al. 2021)** — foundation of modern vision-language.

**Training:** 400M+ image-text pairs. Two encoders (image ViT/ResNet, text transformer). Contrastive objective: correct pairs close, wrong pairs distant. No labels — supervision is which text goes with which image.

**Enables:**
- **Zero-shot classification** — "cat photo" vs "dog photo": embed prompts, embed image, closest wins.
- **Image-text retrieval.**
- **Foundation for VLMs** — image encoder used in many.

**Descendants:**
- **OpenCLIP** — open reproduction with variants.
- **SigLIP (Zhai et al. 2023)** — sigmoid loss instead of contrastive softmax. Better scaling, better performance.
- **EVA-CLIP** — billion+ params.
- **DINOv2** — self-supervised, different training but useful features.

In modern VLMs: vision encoder often CLIP-style (frozen or lightly fine-tuned).

---

## VLM Architectures (`vlm_architectures.py`)

**Three dominant patterns:**

**LLaVA-style (dominant open-source):**
- Frozen/partially-trained CLIP encoder → patch features.
- Small **projection MLP** → LLM embedding space.
- LLM receives image tokens + text tokens as one sequence.

Training: Stage 1 freeze LLM+CLIP, train projection on captions. Stage 2 unfreeze LLM (or LoRA), train on visual instructions.

Simple. Works well. LLaVA, Bunny, MiniCPM-V, many open-source.

**Q-Former (BLIP-2):**
- Image encoder frozen.
- **Q-Former** — small transformer with learnable queries extracts compact features.
- Compact features fed to frozen LLM via cross-attention.

Advantage: fewer image tokens (compression). Disadvantage: needs own training.

**Native multimodal (GPT-4o, Gemini, Chameleon):**
- Pretrained from scratch on interleaved image + text (+ maybe audio/video).
- No separate encoder — everything's tokens.
- Highest quality, most expensive.

**Interleaved cross-attention (Flamingo, IDEFICS):**
- Frozen LLM.
- Vision features fed to cross-attention layers inserted between LLM layers.
- Preserves LLM behavior; adds image conditioning.

**In 2026:** LLaVA-style dominates open-source; native multimodal dominates commercial frontier.

---

## Modern VLMs 2026 (`modern_vlms_2026.py`)

**Commercial:**
- **GPT-4o** — native multimodal.
- **Claude** — text+image → text. Excellent documents.
- **Gemini** — native multimodal + video, very long context.

**Open-source (2025-2026):**
- **Qwen2.5-VL (Alibaba)** — one of best open VLMs. 3B/7B/72B. Strong at docs.
- **Llama-3.2 Vision** — 11B/90B. Solid general.
- **Pixtral** — 12B/larger. Competitive.
- **InternVL 2/3** — very strong.
- **MiniCPM-V** — small, fast, edge.
- **Molmo** — trained on high-quality data, competitive with much larger.

**Specialized:**
- **ColPali** — document retrieval.
- **Donut** — OCR-free document.
- **LayoutLMv3** — text+layout+image fusion.

**Choosing:**
- Enterprise document work: Claude, GPT-4o, or Qwen2.5-VL.
- Cost-sensitive general vision: MiniCPM-V or Llama-3.2 Vision 11B.
- Video: Gemini or Qwen2.5-VL.
- Fine-tunable open: Qwen2.5-VL, Llama-3.2 Vision.

---

## Image Encoding (`image_encoding.py`)

**How images become "tokens" for LLM:**

**1. Preprocessing.** Resize to standard resolution (336², 448², 1024²). Some models (LLaVA-NeXT, Qwen2.5-VL) use **dynamic resolution** — variable size via tiling.

**2. Patch embedding.** Divide into non-overlapping patches (14×14 or 16×16). Each flattened → linear projection → embedding. Result: sequence of patch embeddings (256-1024 per image).

**3. ViT processing.** Patch embeddings + positional. Transformer layers.

**4. Feature projection.** MLP projects to LLM's embedding dim. Sometimes pooling to reduce tokens (Q-Former, cross-attention adapters).

**5. Insertion.** Special tokens delimit position: `<image> [patch tokens] </image>`.

**Cost implications:**
- 336² with 14² patches → 576 tokens per image.
- 1024² → 5000+ tokens per image.
- **High-res is expensive.** Every task should decide the trade-off.

---

## VLM Prompting Patterns (`vlm_prompting_patterns.py`)

**Different from text-only.**

**Grounding** — direct to regions:
> "Look at the top-left quadrant. What color is the object there?"

Much better than open-ended.

**Step-by-step for complex images:**
> "1. Describe what you see. 2. Identify each object. 3. Answer the question."

**Structured output:**
> "Return a JSON with fields: object_name, color, position, count."

Works better than freeform for extraction.

**"If you don't see" instructions:**
> "If the image doesn't clearly show a receipt, respond 'not a receipt'."

Reduces hallucination.

**Multi-image reasoning:**
> "Compare image 1 and image 2. What's different?"

Varies by model. Test.

**Anti-patterns:**
- Vague ("What do you think?") → hallucinates.
- Assuming contents ("Describe the person's mood") → can't reliably assess.
- Fine detail without high resolution.

---

## Visual Grounding (`visual_grounding.py`)

**Grounding = pointing at things in the image.**

**Output types:**
- **Bounding boxes:** `[x0, y0, x1, y1]`.
- **Points:** `(x, y)`.
- **Segmentation masks:** pixel-level.

**Modern VLMs with grounding:**
- **Qwen-VL** — outputs bounding boxes natively.
- **Kosmos-2 (Microsoft)** — grounding-focused.
- **Molmo** — supports pointing.

**Uses:** UI automation (click submit button); document extraction (highlight invoice number); visual robotics; visual search.

**Coordinate conventions vary.** Some pixel coords, some [0, 1000], some [0, 1]. Read the model card.

---

## Document VLM Pipelines (`document_vlm_pipelines.py`)

**Documents are a special multimodal case.**

**Traditional:** PDF → images → OCR → text → LLM. Loses layout, tables, figures.

**VLM-based:** PDF → images → VLM directly. Layout preserved. Higher cost, higher fidelity.

**Hybrid (typical modern):** PDF → images + text extraction. VLM processes images with text as hint. Best of both.

**Specialized:**
- **ColPali** — indexes docs visually, retrieves pages by image, VLM does QA.
- **DocOwl** — document-specific VLM.
- **LayoutLMv3** — text + image + layout fusion.
- **Donut** — end-to-end OCR-free.

**When each:**
- Text-heavy, simple layout → OCR + text LLM (cheapest).
- Layout-critical (forms, tables, figures) → VLM directly.
- Retrieval + QA at scale → ColPali + VLM.

---

## VLM Fine-Tuning (`vlm_finetuning.py`)

**Adds domain vision + instruction adaptation.**

**Common approach:**
- Start with Qwen2.5-VL or Llama-3.2 Vision.
- LoRA on LLM part; sometimes projection layer.
- Vision encoder often frozen (unless heavy domain adaptation).
- Training data: (image, instruction, response).

**Data format (LLaVA-style):**
```json
{
  "id": "...",
  "image": "images/example_001.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is shown in this document?"},
    {"from": "gpt", "value": "This is an invoice from ..."}
  ]
}
```

**LoRA config:** target LLM's attention + MLP; rank 32-64 (larger than text-only — vision-language alignment is harder); LR 1e-4 to 5e-5; 1-3 epochs.

**Data volume:** small (2K-10K) for domain; substantial (20K-100K) for adaptation.

**Frameworks:** Axolotl with multimodal extension; LLaVA training code; Qwen-VL scripts; custom HF+PEFT+Trainer.

**Common failures:** underfit due to too-low rank; image resolution mismatch train/inference; chat template mismatch (VLMs have own image tokens); not freezing vision encoder when should be.

---

## VLM Evaluation (`vlm_evaluation.py`)

**Benchmarks:**
- **MMMU** — college-level multi-discipline.
- **MMBench** — comprehensive skills.
- **MathVista** — visual math.
- **DocVQA** — document QA.
- **ChartQA** — chart understanding.
- **TextVQA** — text in images.
- **RealWorldQA** — practical vision.
- **HallusionBench** — hallucination detection.
- **MMStar** — carefully curated.

**Task-specific:** build your own eval set (200-1000 examples), held out, automated + human sample.

**Gotchas:** position bias (corners worse); resolution effects (match production); language bias (better in English); complexity bias.

---

## VLM Production Patterns (`vlm_production_patterns.py`)

**Different concerns from text-only:**

**Ingestion.** File upload with size/format limits. Malware scan. Format conversion (HEIC → JPG). Resize to model's expected.

**Cost management.** Per-image token count high (500-5000). Cost dominant vs text. Cache image encodings.

**Latency.** Image processing adds 200ms-2s. Prefill dominates TTFT. Streaming still works.

**Safety.** NSFW detection. PII (faces, plates, docs). Deepfake detection. Refuse to describe people.

**Storage.** Media storage expensive. Retention policies. Encrypted at rest.

**Multi-image chat.** Context fills fast. Trimming: drop older images before older text.

**Serving:** vLLM, TGI, SGLang all support VLMs now. Multi-LoRA works for VLMs.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `clip_and_its_descendants.py` | The foundation |
| `vlm_architectures.py` | LLaVA, Q-Former, native, cross-attention |
| `modern_vlms_2026.py` | GPT-4o, Claude, Gemini, Qwen-VL, Llama-Vision |
| `image_encoding.py` | Patches, tokens, projectors |
| `vlm_prompting_patterns.py` | Doing it well |
| `visual_grounding.py` | Pointing at things |
| `document_vlm_pipelines.py` | Documents as multimodal case |
| `vlm_finetuning.py` | LoRA on VLMs |
| `vlm_evaluation.py` | MMMU, MMBench, real tasks |
| `vlm_production_patterns.py` | Latency, cost, safety, storage |

---

*Previous: [← Multimodal Foundations](../multimodal_foundations/README.md) · Next: [Document Intelligence →](../document_intelligence/README.md)*  ·  *Back to [main README](../../README.md)*

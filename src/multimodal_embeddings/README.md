# 🧲 Multimodal Embeddings — Cross-Modal Search

> *If your embedding lives in one space, search only works within one modality. Multimodal embeddings unify text, image, and audio in a shared vector space — the foundation of "search my images with text" and every cross-modal RAG system.*

---

## CLIP Embeddings (`clip_embeddings.py`)

**CLIP produces joint text-image embeddings.**

**How to use:**
```python
text_emb = clip.encode_text("a photo of a red car")
image_emb = clip.encode_image(image)
similarity = text_emb @ image_emb.T
```

**Both are 512-dim (ViT-B/32) or 768-dim (ViT-L/14) or 1024-dim (ViT-H/14) unit vectors.**

**Search patterns:**
- Text → images.
- Image → text.
- Image → images (visual similarity).
- Text → text (CLIP does this too, though not as well as dedicated text embeddings).

**Limitations:**
- Generic image-text training. Domain queries (medical, legal, technical) work worse.
- **Bag of concepts:** recognizes concepts but doesn't reason. "Red car parked next to blue tree" and "blue car parked next to red tree" have similar embeddings.
- Weak on OCR (text within images).
- Weak on fine-grained detail.

**When still the right choice:** general visual search, baseline zero-shot classification, cross-modal RAG entry point.

---

## Sentence-Image Alignment — SigLIP (`sentence_image_alignment.py`)

**SigLIP (Zhai et al. 2023)** — improved CLIP.

**Key change:** replaces CLIP's contrastive softmax loss with sigmoid loss.

**Impact:** better scaling with batch size; better per-image at same param count; cleaner separation of positive/negative pairs.

**In practice:** for new projects, prefer SigLIP over CLIP if quality matters and both available. Same interface.

**SigLIP2 (2024):** further improvements, better semantic understanding.

---

## Multimodal Vector Search (`multimodal_vector_search.py`)

**Storing embeddings in vector DBs.**

**DBs that handle multimodal:** Qdrant (native multi-vector), Weaviate (multimodal features), Milvus (multi-vector), Pinecone (general), pgvector (Postgres extension).

**Indexing strategies:**

**Single unified index:** all items embedded into same space via CLIP/SigLIP. Store: `(id, modality, embedding, metadata)`. Query with any modality → results across all.

**Separate indexes per modality:** text via text-only encoder (BGE, E5); images via CLIP. Query fanned to both; results fused.

**Hybrid multi-vector:** each item has multiple embeddings. Multi-vector retrieval, late fusion.

**Which:**
- Single: simple, works for basic cross-modal RAG.
- Separate + fusion: better when text-only quality matters (long passages).
- Multi-vector: best quality, most complex.

---

## Cross-Modal Retrieval (`cross_modal_retrieval.py`)

**Given query in one modality, retrieve items in another.**

**Text → image:** "dogs playing in snow" → find matching images. Standard CLIP/SigLIP.

**Image → text:** upload image → find related documents. Embed image, search over text embeddings.

**Image → image:** visual similarity. E-commerce reverse search.

**Text → video clips:** "clips of CEO talking about Q4." Video frame + audio transcript embeddings.

**Multi-modal query:** text + reference image → find images matching both. Weighted sum of query embeddings.

**Metrics:**
- **Recall@K** — is correct item in top K?
- **MRR** — how highly ranked?
- **NDCG** — graded relevance.

**Rule:** cross-modal quality is domain-dependent. Test on your actual query distribution before deploying.

---

## Unified Multimodal Index (`unified_multimodal_index.py`)

**Everything in one vector space.**

**Design:** all content types embedded via multimodal model (CLIP, SigLIP, ImageBind). Single vector space; single index. Metadata differentiates modality for post-processing.

**Pros:** simple architecture; one query cross-modal natively; cache-friendly.

**Cons:** embedding quality per modality limited by joint model. Very long text loses quality (CLIP trained on short captions). Domain-specific quality capped at model's training.

**Alternative: modality-specific + bridging layer.** Each modality embedded with its best model. Learned linear projection maps to shared space. Better quality per modality; more training work.

---

## Multimodal RAG Architecture (`multimodal_rag_architecture.py`)

**End-to-end:**
```
User Query (text, image, or both)
    ↓
Query embedding (multimodal-capable encoder)
    ↓
Retrieval
    ├── Text embeddings (specialized text encoder)
    ├── Image embeddings (CLIP/SigLIP)
    ├── Video embeddings (ColPali-style or frame-based)
    └── Audio embeddings (audio LLM encoders)
    ↓
Fusion / reranking (across modalities)
    ↓
Top-K items (may be mixed modalities)
    ↓
Generation (VLM if images present, else LLM)
    ├── Grounded on retrieved multimodal content
    ├── Citations back to source
    └── Refusal if insufficient evidence
    ↓
Response
```

**Design considerations:**
- **Modality-aware retrieval** — text queries prefer text; visual queries prefer images.
- **Modality-aware reranking** — different reranker per modality.
- **Fusion strategy** — weighted (favor high-signal), reciprocal rank fusion, or learned.
- **Generation model** — VLM required if images in context.

**Real-world example:** enterprise assistant answering "show me the graph from Q4 report" — retrieval finds specific chart image within specific document; VLM extracts data from chart; response includes chart image + numerical answer.

**Complexity:** multimodal RAG significantly more complex than text RAG. Storage cost, latency, cost/query all increase. Reserve for use cases where multimodal actually matters.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `clip_embeddings.py` | Text ↔ image |
| `sentence_image_alignment.py` | SigLIP |
| `multimodal_vector_search.py` | Vector DBs, indexing strategies |
| `cross_modal_retrieval.py` | Query in one modality |
| `unified_multimodal_index.py` | Everything in one space |
| `multimodal_rag_architecture.py` | End-to-end |

---

*Previous: [← Video Understanding](../video_understanding/README.md) · Next: [Multimodal Generation →](../multimodal_generation/README.md)*  ·  *Back to [main README](../../README.md)*

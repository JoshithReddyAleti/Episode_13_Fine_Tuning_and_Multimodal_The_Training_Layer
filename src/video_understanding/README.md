# 🎥 Video Understanding — Video-Native Models

> *Video is images + time + audio. Every dimension adds cost. Every dimension adds capability. Understanding video is where multimodal AI meets real bandwidth and storage constraints.*

---

## Video Representation (`video_representation.py`)

**Video = frames + audio track (usually).**

**Frame representation:** extract N frames per second (typically 1-4 fps for understanding — full 30 fps rarely necessary). Each frame treated as image, encoded like VLM does.

**Temporal representation:** frames alone lose temporal information. Options:
- **Frame-level features + temporal pooling** — average over time.
- **Temporal transformers** — attention across frame features.
- **Time-embeddings** — positional for time as well as space.
- **Motion features** — optical flow between adjacent frames.

**Cost:** 1 minute @ 1 fps @ 576 tokens/frame = **34,560 tokens**. 10 min = **345,600 tokens** — most models can't handle. Compression strategies matter.

**Common compressions:** keyframe extraction (visually distinct); adaptive sampling; frame subsampling (0.5 fps); token compression (Q-Former-style).

**Audio track:** handled separately (see audio section). Some models incorporate audio jointly; most process audio via ASR separately.

---

## Video LLMs (`video_llms.py`)

**Models that natively understand video.**

**Architecture patterns:**

**Frame VLM + temporal pooling (baseline):** each frame → VLM features. Simple pooling or attention over frames. Fed to LLM.

**Spatiotemporal transformer:** video tokens carry both spatial (patch) and temporal (frame) position. Full attention across all tokens. Expensive but expressive.

**Q-Former with temporal queries (Video-LLaMA, VideoChat):** Q-Former compresses frames. Temporal queries extract time-varying features.

**Sparse temporal attention:** only attend across subset of frames. Scales to longer videos.

**Notable video LLMs (2025-2026):**
- **Gemini** — native video via long context. Can process hours.
- **GPT-4o** — video understanding.
- **Video-LLaMA / VideoChat / VideoChat2** — early open-source.
- **LLaVA-NeXT-Video** — LLaVA family with video.
- **Qwen2-VL** — supports video natively.
- **InternVideo 2.5** — strong open-source.
- **Video-Salmonn** — audio + video joint.

**Real-world capabilities:**
- Short video QA (10-60s): reliable.
- Medium video (1-10 min): decent with good models.
- Long video (>10 min): challenging; needs specialized handling.

---

## Temporal Reasoning (`temporal_reasoning.py`)

**Understanding change over time is harder than static images.**

**Types:**
1. **Sequence of events.** What happened first, then next?
2. **Duration.** How long did X last?
3. **Causation.** Why did X happen?
4. **Change detection.** What's different between now and then?
5. **Motion and dynamics.** Object trajectory, movement patterns.

**Failure modes:**
- Frame skipping hides critical events.
- Static-image bias — model treats each frame independently, loses temporal thread.
- Temporal hallucination — model makes up "then" claims not in frames.

**Improvement:** explicit timestamps in inputs/outputs; chain-of-thought for temporal analysis; retrieve relevant frames rather than uniform sample.

---

## Long Video Understanding (`long_video_understanding.py`)

**Videos beyond a few minutes are the hardest problem.**

**Strategies:**

**Hierarchical summarization:** chunk video (1-min chunks). Summarize each with smaller model. Aggregate summaries. Query against hierarchical index.

**RAG over video chunks:** chunk → embed → retrieve relevant chunks per query → VLM answers on retrieved.

**Long-context native:** feed entire video (heavy subsampling) to long-context model. Gemini can handle hours at high cost.

**Specialized long-video models:** LongVA, LongVILA, LongViT — research models focused on long-video. Efficient temporal attention.

**Practical patterns:**
- **Preview + drill-down** — overview via summary; retrieve specific segments for detailed QA.
- **Timestamp-first** — extract event timestamps, answer about specific windows.
- **Audio-first** — ASR the whole video, RAG over transcript; use video for grounding.

**Cost:** long video is 10-100× the cost of short. Batch processing, caching, smart retrieval essential.

---

## Video Search and Retrieval (`video_search_and_retrieval.py`)

**Multimodal RAG on video.**

**Indexing:**
1. Extract keyframes (visual novelty detection).
2. For each keyframe, compute embedding (CLIP/SigLIP).
3. Transcribe audio, segment by time.
4. Embed audio segments.
5. Store: (video_id, timestamp, visual_embedding, transcript_segment, transcript_embedding).

**Retrieval:**
- Text query → embed → find nearest visual + transcript embeddings.
- Fuse results (rank fusion).
- Return top-K (video, timestamp) results.

**Combined example:**
- Query: "show me the demo of feature X."
- Visual retrieval finds screens showing feature.
- Transcript retrieval finds mentions of feature name.
- Fusion: rank videos where both signals agree.

**Fine points:** deduplicate overlapping windows (frames 10, 12, 14 → one result); boundary detection (extend match to natural clip); metadata filtering (time, creator, tags).

**Use cases:** enterprise video search; media libraries; meeting recall ("when did we discuss X?"); e-commerce (products in influencer videos).

---

## Video QA (`video_qa.py`)

**End-to-end video question answering.**

**Pipeline:**
1. **Query analysis.** Whole-video or specific-moment question?
2. **Retrieval / segmentation.** If specific, retrieve relevant clips. If whole, prepare full video representation.
3. **VLM inference.** Video LLM answers.
4. **Post-processing.** Add timestamps, clip citations.

**Question types and approaches:**

| Question type | Approach |
|---|---|
| "What is happening?" | Whole-video processing |
| "When does X happen?" | Retrieval + timestamp extraction |
| "How many times does Y occur?" | Full pass with counting |
| "Compare A and B" | Retrieve both, then compare |
| "Summarize this video" | Hierarchical summarization |
| "Why did X happen?" | Retrieve preceding context, causal reasoning |

**Metrics:** answer accuracy (human-judged); timestamp accuracy; retrieval recall; cost per query.

---

## Video Production Pipelines (`video_production_pipelines.py`)

**Handling video at scale.**

**Preprocessing:**
- **Ingestion:** file upload, size limits, format validation.
- **Transcoding:** to standard format (H.264 MP4).
- **Frame extraction:** at target fps.
- **Audio extraction:** separate track.
- **Metadata:** duration, resolution, codec, timestamps.

**Storage:**
- Video files: object storage.
- Frame images: cached separately.
- Audio: transcoded for ASR.
- Embeddings: vector DB.
- Metadata: relational DB or search index.

**Compute pipeline:**
- Async queue (uploads trigger processing).
- GPU workers for VLM/ASR/embedding.
- Retry logic (transient failures).
- Status tracking (when queryable).

**Cost management:**
- **Sampling** — aggressive frame subsample for long/low-value.
- **Caching** — embeddings permanent.
- **Tier by value** — expensive full-VLM only for high-value.
- **Batch processing** — off-peak GPU utilization.

**Latency SLAs:** upload → queryable 30 seconds to 5 minutes. Query response 1-10 seconds.

**Failure modes:** corrupt videos; audio-video sync; very long videos (memory pressure); streaming vs static handling.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `video_representation.py` | Frames, tokens, temporal |
| `video_llms.py` | Video-LLaMA, Gemini, Qwen2-VL |
| `temporal_reasoning.py` | Sequence, duration, causation |
| `long_video_understanding.py` | Beyond a few minutes |
| `video_search_and_retrieval.py` | Multimodal RAG on video |
| `video_qa.py` | End-to-end question answering |
| `video_production_pipelines.py` | Preprocessing, storage, serving |

---

*Previous: [← Audio and Speech](../audio_and_speech/README.md) · Next: [Multimodal Embeddings →](../multimodal_embeddings/README.md)*  ·  *Back to [main README](../../README.md)*

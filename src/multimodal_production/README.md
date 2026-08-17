# 📦 Multimodal Production — Shipping Multimodal Systems

> *A multimodal demo is a Jupyter notebook. A multimodal production system is media pipelines, cost management, safety filters, and observability that most teams underestimate by 5×.*

---

## File Upload Pipelines (`file_upload_pipelines.py`)

**Requirements:** large file support (videos can be GB); progress reporting; chunked/resumable uploads (unreliable networks); format validation (reject invalid/unsupported early); malware scanning; access control.

**Patterns:**

**Direct-to-storage upload:** client uploads directly to S3/GCS via presigned URL. Server never sees file transit. Best for large files.

**Multipart via API:** client uploads to API. API validates, forwards to storage. More control, more bandwidth.

**Chunked resumable (tus.io, S3 multipart):** break upload into chunks. Retry failed chunks without restart. Essential for video files.

**Post-upload workflow:**
1. File arrives in staging bucket.
2. Validation (format, size, malware).
3. Move to processing queue.
4. Metadata written to DB (status: "processing").
5. Async workers pick up.
6. Status updated as processing completes.
7. Media becomes queryable.

---

## Media Preprocessing (`media_preprocessing.py`)

**Everything before media hits the model.**

**Images:** resize to model's expected input; format conversion (HEIC → JPG for iOS uploads); EXIF stripping (remove PII: GPS, timestamps, camera); orientation correction; color space normalization (sRGB).

**Video:** transcode to standard format (H.264 MP4); frame extraction at target FPS; keyframe detection; audio track extraction; resolution normalization (downsize very high-res for cost).

**Audio:** resample to model's expected rate (16 kHz for speech); mono conversion; loudness normalization; format conversion (WAV/FLAC internal, MP3/AAC external).

**Documents:** PDF page rasterization; text layer extraction in parallel; split by page for parallelization.

**Tools:** FFmpeg (video/audio Swiss Army knife); Pillow, opencv-python (image); PyMuPDF, pdfplumber (PDF); librosa, torchaudio (audio).

**Cache preprocessed outputs.** Reprocessing many times is wasteful.

---

## Multimodal Caching (`multimodal_caching.py`)

**Media expensive to process, easy to cache.**

**Levels:**

**File hash → preprocessed output.** SHA-256 the uploaded file. Cache preprocessed version. Same file uploaded twice → hit cache.

**Preprocessing output → embedding.** Preprocessed image → embedding cached. Preprocessed audio → transcript cached. Preprocessed video → per-frame features cached.

**Content + query → result.** Full inference result cached by (content_hash, query_hash). Especially valuable when same media queried many times.

**Deduplication across users.** Same file uploaded by different users? Store once (careful access control). Common for stock images, memes, reference docs.

**Cache eviction:** LRU with time-based expiration. Preserve high-value items.

**Storage tiers:** hot (Redis/Memcache); warm (object storage with indexed lookup); cold (Glacier for retention).

**Cache hit rates typical:** file-level 20-40% enterprise (many users share reference docs); query-level 10-30% consumer, higher enterprise.

**Real ROI:** well-designed cache reduces media processing costs by 30-70%.

---

## Multimodal Cost Management (`multimodal_cost_management.py`)

**Cost drivers:**

**Storage.** Media files large. Cache tiers, retention policies matter. Cold storage for rarely-accessed originals.

**Compute:** VLM inference $0.001-0.05/image; video processing $0.10-5/min; audio transcription $0.001-0.02/min; embedding generation $0.0001-0.001/item.

**Bandwidth.** Uploads and downloads at scale. CDN costs for consumer apps.

**Third-party APIs.** OpenAI, Anthropic, Google multimodal APIs at cents-to-dollars per request.

**Optimization tactics:**

**Right-size model.** Smaller VLMs where sufficient. Reserve big models (GPT-4o, Claude) for hard cases. Route by task complexity.

**Reduce input size.** Downsample images/video before inference where quality allows. Extract only relevant regions. Frame-sample video aggressively.

**Cache aggressively.**

**Batch processing.** Off-peak batch for non-urgent. Amortize model warm-up.

**Progressive processing.** Fast/cheap first pass for relevance. Only expensive processing on relevant items.

**Cost attribution.** Tag every job with tenant, user, endpoint. Per-tenant tracking. Cost dashboards to detect anomalies.

**Budgets:** per-tenant monthly caps. Alerts at 50%, 80%, 100%. Graceful degradation when exceeded.

---

## Multimodal Observability (`multimodal_observability.py`)

**Extends Episode 11 with media-specific concerns.**

**Additional metrics:**
- Per-modality latency — video processing time, audio transcription, image VLM inference.
- File size distributions — track ingestion.
- Processing queue depth — media processing can back up.
- Cache hit rates — critical cost signal.
- Media storage growth — capacity planning.
- Format distribution — detect new formats needing handling.
- Failure rates by media type — corrupt PDFs, unsupported video codecs.

**Distributed tracing:** trace request from upload through preprocessing through inference. Each stage adds spans. Identify bottlenecks.

**Sampling for cost:** full logging expensive at media scale. Sample: log 1-10% fully; aggregate metrics for rest.

**Media-specific alerts:** preprocessing failure spike (bad uploads or pipeline bug); VLM P99 exceeds threshold; cost/query exceeds threshold; storage growth exceeds forecast.

---

## Multimodal Safety (`multimodal_safety.py`)

**Extra concerns beyond text.**

**NSFW / adult content:** input (upload screening) and output (generation screening). Classifiers: NudeNet, Google SafeSearch, commercial (AWS Rekognition, Azure Content Moderator).

**Violent content:** similar detection. Region-specific thresholds.

**PII in media:**
- **Faces** — face detection + blurring.
- **License plates** — plate detection + blurring.
- **Text in images** — OCR + PII detection (SSNs, credit cards).
- **Voices** — speaker identification.
- **Documents** — detect confidential markings, PII fields.

**CSAM (Child Sexual Abuse Material):** zero tolerance. Automated detection required by law in many jurisdictions. Hash-based (NCMEC's PhotoDNA) + classifier-based. Report + hard block.

**Deepfake detection:** emerging area. Watermarks in generated content (SynthID from Google, etc.). Classifiers for likely-synthetic.

**Copyright / IP:** reverse image search against copyrighted material. Music fingerprinting for uploaded audio. Text plagiarism detection.

**Output moderation for generative:** every generated image/video/audio passes safety classifiers before returning to user. Refusal preferable to unsafe output.

---

## Multimodal Serving Stacks (`multimodal_serving_stacks.py`)

**Reference architectures.**

**Consumer app (voice + image chat):**
```
Mobile client → API gateway (auth, rate limit)
              → Media processing service (transcode, preprocess)
              → Model serving (VLM + Whisper + TTS)
              → CDN for returning generated media
```

**Enterprise document intelligence:**
```
Document ingestion → S3
    → Async processing workers
    ├── PDF text extraction
    ├── OCR (for scans)
    ├── Layout analysis
    ├── Table extraction
    └── VLM for complex pages
    → Embedding + vector DB indexing
    → Query API (RAG over multimodal index)
    → Answer with citations
```

**Media library / search:**
```
Upload → object storage
    → Preprocessing (frame extract, transcribe, embed)
    → Vector DB (visual + text embeddings)
    → Search API (cross-modal)
    → Playback / preview UI
```

**Voice agent (production):**
```
Telephony / WebRTC gateway
    → Audio streaming → ASR (streaming Whisper)
    → Turn detection (VAD)
    → LLM (agent + tools)
    → Streaming TTS
    → Audio out
    → Call recording + analytics
```

**Common infrastructure:** model gateway (Episode 12) to abstract underlying models; object storage for media; vector DB for embeddings; traditional DB for metadata; queue for async processing; Kubernetes for stateless services; GPU workers for inference.

**Rule:** every multimodal production system needs each of: media handling, cost management, safety, observability. Skipping any leads to failure modes that surface expensively in production.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `file_upload_pipelines.py` | Handling media at scale |
| `media_preprocessing.py` | Resize, transcode, chunk |
| `multimodal_caching.py` | File hashing, dedup |
| `multimodal_cost_management.py` | Media is expensive |
| `multimodal_observability.py` | Extending Episode 11 |
| `multimodal_safety.py` | NSFW, PII, CSAM, deepfakes |
| `multimodal_serving_stacks.py` | Full architecture |

---

*Previous: [← Multimodal Generation](../multimodal_generation/README.md)*  ·  *Back to [main README](../../README.md)*

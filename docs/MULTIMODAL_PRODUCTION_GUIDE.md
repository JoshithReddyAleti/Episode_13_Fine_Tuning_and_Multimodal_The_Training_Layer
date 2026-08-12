# Multimodal Production Guide

## The 6 pillars

1. **File upload** — chunked, resumable, validated, scanned.
2. **Preprocessing** — resize, transcode, normalize per modality.
3. **Caching** — file hash → preprocessed, embeddings, results.
4. **Cost management** — right-size models, cache aggressively, batch off-peak.
5. **Observability** — per-modality latency, queue depth, cache hit rates.
6. **Safety** — NSFW, PII, CSAM, deepfakes, IP.

## Cost drivers

- Storage (media is large).
- Compute (VLM $0.001-0.05/image, video $0.10-5/min, audio $0.001-0.02/min).
- Bandwidth (uploads/downloads, CDN).
- Third-party APIs.

## Cache hit rates you should see

- File-level: 20-40% in enterprise (users share reference docs).
- Query-level: 10-30% consumer, higher enterprise.

## Safety filters

- NSFW: NudeNet, cloud APIs (Rekognition, Content Moderator).
- PII: face/plate blurring, OCR + PII detection on documents.
- CSAM: hash-based (PhotoDNA) + classifier. Legal reporting required.
- Deepfakes: SynthID and similar watermarks, detection classifiers.

## Serving reference architectures

- Consumer app (voice + image chat).
- Enterprise document intelligence.
- Media library / search.
- Voice agent (telephony integration).

Every one needs each of the 6 pillars. Skipping any leads to expensive production failures.

# 🎨 Multimodal Generation — Not Just Understanding

> *Consuming multimodal content is one thing. Producing it is another discipline entirely — different models, different economics, different safety concerns.*

---

## Image Generation APIs (`image_generation_apis.py`)

**The 2026 landscape:**

**Commercial APIs:**
- **DALL-E 3 (OpenAI)** — text-to-image, integrated with ChatGPT.
- **Midjourney** — highest artistic quality, Discord-first, API in beta.
- **Ideogram** — strong text-in-image, good general.
- **Adobe Firefly** — commercial-safe (licensed training data).
- **Flux (Black Forest Labs)** — open weights, top quality.

**Open-source (self-hosted):**
- **Stable Diffusion 3 / 3.5** — Stability's flagship.
- **Flux.1 dev / schnell** — top open-source; commercial license considerations.
- **SDXL** — older but widely deployed, many fine-tunes.
- **PixArt-alpha, Kandinsky, Kolors** — alternatives.

**Model characteristics:** diffusion-based (iterative denoising from noise); latent diffusion (compressed latent space for efficiency); transformer-based diffusion (DiT) — newer, better scaling.

**Cost (2026):** commercial APIs $0.02-0.08/image; self-hosted $0.001-0.01/image on H100.

**Latency:** fast models (Flux schnell, SDXL Turbo) 1-3s; standard 3-10s; high-quality 10-30s.

**Choosing:**
- Prototype: commercial API.
- High volume: self-hosted with quality tuning.
- Specific style: fine-tuned open-source.
- Commercial-safe (no copyright risk): Firefly, or self-hosted vetted training data.

---

## Prompting Image Models (`prompting_image_models.py`)

**Image gen prompting is a craft.**

**General principles:**
- **Subject.** What is the main thing? Be specific.
- **Style.** Photography, painting, illustration, 3D render.
- **Composition.** Angle, framing, distance.
- **Lighting.** Warm, cool, dramatic, natural.
- **Details.** Textures, colors, secondary elements.
- **Negative prompts.** What to avoid.

**Example structure:**
```
"[subject], [action/pose], [environment], [style], [lighting],
[composition], [details], [modifiers]

negative: blurry, distorted, extra limbs, low quality"
```

**Model-specific quirks:**
- **Midjourney:** style-heavy, artistic focus. `--ar`, `--stylize`.
- **DALL-E:** long natural-language prompts work well.
- **SD/Flux:** shorter tag-style + natural-language mix; ControlNet for structure.

**Common failures:** text in images (Flux, Ideogram best); hands and fingers (historical weakness, improving); precise counting; complex compositions with multiple subjects and specific positions.

---

## Controlled Generation (`controlled_generation.py`)

**Beyond text prompts — steering with additional signals.**

**ControlNet (Zhang et al. 2023):** extra conditioning (edge maps, depth maps, poses, segmentation). Frozen base diffusion + trainable ControlNet branch. "Generate this scene with this exact composition."

**IP-Adapter:** image prompting — use reference image as style/content guide. Preserves style/identity.

**InstantID / PhotoMaker:** preserve specific person's identity across outputs. Ethics: consent, deepfakes, impersonation.

**LoRA fine-tuning (visual):** fine-tune small adapter for specific style/character/subject. Community models (Civitai, HuggingFace) share LoRAs extensively.

**Combining controls:** base prompt + ControlNet pose + IP-Adapter for style + subject LoRA = fully-controlled generation. Composable but careful weighting needed.

**Where controlled generation matters:** brand-consistent imagery; product visualization; character consistency (comics/animation); design iteration (same composition, different styles).

---

## Image Editing (`image_editing.py`)

**Not just generating from scratch — modifying existing.**

**Inpainting:** mask region, fill with generated content that fits context. Uses: object removal, replacement, background fill. Standard in SD, Flux, DALL-E.

**Outpainting:** extend image beyond borders. Uses: aspect ratio change, panorama, zoom-out.

**Instruct-based editing:** natural language commands. Models: InstructPix2Pix, MagicBrush, AnyEdit. "Make the sky more dramatic" / "add a red hat."

**Style transfer:** apply style of one image to content of another. VLM-based approaches becoming dominant.

**Production workflows:** real estate (virtual staging); e-commerce (product on backgrounds); content creation (rapid hero image iteration).

**Quality considerations:** edit locality (unrelated regions unchanged); consistency (multiple edits compose); realism.

---

## Video Generation (`video_generation.py`)

**Text-to-video and image-to-video.**

**Commercial:**
- **Sora (OpenAI)** — high-fidelity, up to 60 seconds.
- **Runway Gen-3** — commercial-grade, popular in creative industry.
- **Kling (Kuaishou)** — Chinese offering, very strong.
- **Luma Dream Machine** — accessible, decent quality.
- **Pika** — commercial video gen.

**Open-source:**
- **Stable Video Diffusion** — open, image-to-video.
- **CogVideoX** — open, text-to-video.
- **Mochi (Genmo)** — recent open-weight.
- **HunyuanVideo (Tencent)** — top open-source video gen 2024-2026.

**Characteristics:** duration 2-10 seconds typical (some reach 60s); resolution 480p-1080p; FPS 24-30; cost $0.5-5 per generated clip on commercial APIs.

**Failure modes:** object permanence (objects morph or disappear mid-clip); physics (unnatural motion); text in video (extremely hard); fine control limited.

**Use cases (2026):** marketing/social; storyboarding; concept previsualization; educational; advertising.

---

## Audio Generation (`audio_generation.py`)

**Beyond TTS — music, sound effects, ambient audio.**

**Music:**
- **Suno v3+** — text-to-music with lyrics. Highest quality general music AI.
- **Udio** — competing platform.
- **MusicGen (Meta)** — open-source.
- **Stable Audio (Stability)** — open, sound effects and music.

**Sound effects:**
- **AudioGen (Meta)** — open-source.
- **ElevenLabs Sound Effects** — commercial.
- Text-to-audio ("sound of a dog barking, then footsteps on gravel").

**Voice (expressive TTS):** modern TTS blurs into audio generation with emotion and style control (see audio section).

**Cost:** music $0.1-0.5/song (commercial), lower self-hosted; sound effects $0.01-0.10/clip.

**Ethical/legal:** training data provenance (many music AIs train on copyrighted material); attribution; rights ownership. Legal landscape rapidly evolving.

---

## Multimodal Agents (`multimodal_agents.py`)

**Agents that generate media as part of their action space.**

**Example capabilities:**
- **Content creation agent** — topic → blog post + hero image + social clips.
- **Product photography agent** — uploaded product → variants on backgrounds, angles, lifestyles.
- **Video editing agent** — raw meeting recording → highlights, chapter markers, thumbnail.
- **Multimodal report agent** — data → written report + generated charts + hero visuals.

**Architecture:** base agent (Episode 9) with tools for each generation modality. Tools: `generate_image`, `edit_image`, `generate_video`, `generate_music`, `generate_voiceover`. Agent plans sequence; tools execute. Human review checkpoint for high-value outputs.

**Design considerations:** cost budgets (media generation expensive; agent cost-aware); fallback (if generation fails or quality low, retry or degrade); style consistency (across the workflow); safety (content moderation on all generated media).

**Real-world adoption:** growing in creative industries, marketing, educational content. Still early for enterprise workflows.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `image_generation_apis.py` | DALL-E, Midjourney, SD, Flux |
| `prompting_image_models.py` | The prompting craft |
| `controlled_generation.py` | ControlNet, IP-Adapter, LoRA |
| `image_editing.py` | Inpainting, outpainting, instruct |
| `video_generation.py` | Sora, Runway, Kling |
| `audio_generation.py` | Music, sound effects |
| `multimodal_agents.py` | Agents that generate media |

---

*Previous: [← Multimodal Embeddings](../multimodal_embeddings/README.md) · Next: [Multimodal Production →](../multimodal_production/README.md)*  ·  *Back to [main README](../../README.md)*

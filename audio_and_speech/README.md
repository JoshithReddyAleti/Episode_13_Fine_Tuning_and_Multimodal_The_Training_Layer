# 🎙️ Audio and Speech — Whisper and Beyond

> *Voice is the most natural human interface. Voice agents are the most-shipped multimodal product in 2026. This section is the audio stack from waveform to shipped voice UI.*

---

## Audio Representation (`audio_representation.py`)

**Waveform:** raw amplitude samples. 16 kHz mono = 16,000 samples/sec = 32 KB/sec.

**Mel Spectrogram:** time-frequency representation, mel-warped. Standard input for Whisper. E.g. 80 bins × 3000 frames for 30 seconds.

**Discrete audio tokens (VQ-based):** audio compressed to codebook indices. Enables audio LLMs treating audio as tokens. EnCodec, SoundStream, WavTokenizer.

**Neural audio embeddings:** learned continuous. Whisper's encoder produces such embeddings.

**Practical:** mel spectrogram dominates for ASR. Discrete tokenization emerging for audio LLMs.

---

## Whisper Deep Dive (`whisper_deep_dive.py`)

**Whisper (Radford et al. 2022)** — the model that transformed speech recognition.

**Architecture:** encoder-decoder transformer. Encoder processes mel spectrogram → contextualized audio features. Decoder produces text conditioned on audio. Multi-task: transcription, translation, language ID, VAD, timestamps.

**Training:** 680,000 hours weakly-supervised multilingual audio.

**Sizes (2026):**
- **tiny (39M):** fast, mediocre.
- **base (74M):** good quality-per-cost.
- **small (244M):** solid.
- **medium (769M):** near-frontier for many uses.
- **large-v3 (1.5B):** state-of-the-art open-source.
- **turbo (809M):** large-v3 variant, speed-optimized.

**Which:**
- Real-time streaming: tiny or base.
- Batch high-quality: large-v3 or turbo.
- Multilingual: large-v3 best coverage.

**Serving:** faster-whisper (CTranslate2, 4× faster than reference); whisper.cpp (CPU/M-series); WhisperX (+ diarization); vLLM now supports Whisper.

**Cost:** ~$0.001-0.01/min self-hosted. Managed APIs: $0.006-0.02/min.

---

## Streaming ASR (`streaming_asr.py`)

**For voice agents and live captions.**

**Approaches:**

**Sliding window:** buffer 5-10s, transcribe on new audio, dedup overlaps.

**Chunk-based with context:** fixed chunks with overlap. Reprocess boundaries.

**True streaming models:** NeMo Streaming Conformer (NVIDIA); RNN-T (natively streaming); Whisper streaming variants (hacks for lower latency).

**Latency budgets:**
- Voice agent: sub-500ms speech end to output.
- Live captions: 1-2s acceptable.
- Meeting transcription: 3-10s.

**Trade-off:** streaming introduces revisions (partial hypotheses updated). UI must show tentative text updates.

---

## Speaker Diarization (`speaker_diarization.py`)

**Who said what in multi-speaker audio.**

**Diarization pipeline:** VAD → speaker embedding per segment → cluster → speaker count estimation.

**End-to-end diarization:** models jointly detect speech and identify speakers. Higher quality, harder to train.

**Tools:** pyannote.audio (open); NeMo Diarization (NVIDIA); WhisperX (diarization + Whisper); AWS Transcribe, Azure Speech, Google Speech-to-Text.

**Real-world:**
- 2-speaker interviews: 90%+ correct.
- 3-4 speaker meetings: 70-85%.
- 5+ with overlap: 50-70%.

**Improving:** pre-enrolled speaker embeddings (if speakers known); source separation before diarization; manual post-editing for high-value.

---

## Voice Activity Detection (`voice_activity_detection.py`)

**VAD** = detecting speech vs silence.

**Essential for:** segmenting long audio; turn-taking in voice agents; reducing transcription cost.

**Modern VAD:** **Silero VAD** (open, fast, high quality); WebRTC VAD (old, ubiquitous); neural VAD.

**Latency:** typically <30ms — fast enough real-time.

**Use in voice agents:** detect speech onset → start streaming ASR; detect silence → mark turn end; prevent talking over user.

---

## Audio LLMs (`audio_llms.py`)

**LLMs taking audio input directly, without transcribing first.**

**Why:** preserves prosody, emotion, tone lost in transcription; handles overlapping speech better; enables audio understanding beyond text.

**Architectures:**

**Audio encoder + LLM:** Whisper-encoder-like → embeddings → projection → LLM's embedding space.

**Native audio-in:** GPT-4o (also produces audio); Gemini; Qwen2-Audio (open); SALMONN (open).

**Capabilities:** speech understanding (better than ASR+LLM); sound event recognition; music understanding; emotion detection; speaker characteristics.

**Trade-off:** much larger and slower than Whisper. For pure transcription, ASR+LLM more efficient. For rich audio understanding, audio LLMs worth it.

---

## TTS Systems (`tts_systems.py`)

**Text-to-Speech: text → natural voice.**

**Neural TTS (voice cloning capable):**
- **XTTS v2 (Coqui)** — open, multilingual, cloning.
- **Bark (Suno)** — open, expression control.
- **StyleTTS 2** — open, high quality.
- **Piper** — open, fast, edge.

**Commercial APIs:**
- **ElevenLabs** — best-in-class, cloning.
- **PlayHT** — competitive, wide languages.
- **OpenAI TTS** — good, integrated.
- **Google/Azure/AWS** — cloud incumbents.

**Quality metrics:** MOS 1-5. Modern TTS: 4.0-4.7.

**Latency:** streaming TTS starts audio while generating (sub-500ms first audio); non-streaming waits for full audio.

**Cost:** open-source $0.001-0.01/1000 chars; cloud $0.02-0.30/1000 chars.

**Use cases:** voice agents (streaming, low latency); audiobook (non-streaming, quality focus); accessibility; dubbing.

---

## Voice Cloning and Ethics (`voice_cloning_and_ethics.py`)

**Voice cloning:** given few seconds of someone's voice, produce arbitrary speech in that voice.

**Capabilities (2026):** 3-10 seconds of clean reference → highly convincing clone; cross-lingual (clone in English, generate in Spanish with same voice); emotion transfer.

**Uses:** audiobook with author's voice; localization/dubbing; restoration (accessibility, memorial); personalized assistants.

**Abuse potential:** deepfakes for fraud; disinformation; non-consensual content; identity theft.

**Responsible practices:**
- **Consent** before cloning any real person.
- **Watermarking** — inaudible signals identifying synthetic.
- **Detection** — classifiers for synthetic speech.
- **Access controls** — verified identity for cloning APIs.
- **Legal** — EU AI Act, evolving US legislation.

**Rule for engineers:** never clone without documented consent. Consider watermarking. Watch legal developments.

---

## Voice Agents Architecture (`voice_agents_architecture.py`)

**End-to-end stack:**
```
User mic → VAD → Streaming ASR → Turn-end detection
→ Full transcript → LLM (agent + tools) → Response text
→ Streaming TTS → User speakers
```

**Latency budget for natural conversation:**
- End-of-speech to first-audio: **<800ms** natural, <1500ms tolerable.
- Split: VAD end-of-turn 200-400ms; ASR final 100-200ms; LLM first token 100-300ms; TTS first audio 100-200ms.

**Optimization:** speculatively start LLM on partial ASR; pre-warm TTS with filler word; cache common responses; overlap TTS generation with audio playback.

**Interruption handling:** detect user speech during TTS playback; halt TTS immediately; buffer partial LLM output as context; resume or restart.

**Frameworks/platforms:** LiveKit Agents; Pipecat (Daily); Vapi; Retell; custom.

**Production requirements:** telephony (SIP, WebRTC); call recording/compliance; multi-lingual switching; cost per minute tracking; escalation to human path.

---

## Audio Evaluation (`audio_evaluation.py`)

**ASR:**
- **WER (Word Error Rate)** — dominant. Lower is better.
- **CER** — for languages without word boundaries.
- **RTF (Real-Time Factor)** — processing time / audio duration. <1.0 = real-time.

**TTS:**
- **MOS** — human 1-5.
- **CMOS** — pairwise preference.

**Voice agents (end-to-end):**
- End-to-end latency.
- Turn success rate.
- User satisfaction (CSAT).
- Task completion rate.
- Cost per turn/call.

**Diarization:**
- **DER (Diarization Error Rate).**
- **JER (Jaccard Error Rate).**

**Rule:** benchmark against real production audio (noise, accents, overlapping speech, disfluencies) — not clean studio data.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `audio_representation.py` | Waveform, spectrogram, tokens |
| `whisper_deep_dive.py` | Architecture, sizes, serving |
| `streaming_asr.py` | Real-time transcription |
| `speaker_diarization.py` | Who said what |
| `voice_activity_detection.py` | VAD in the pipeline |
| `audio_llms.py` | Audio-in LLMs |
| `tts_systems.py` | XTTS, Bark, ElevenLabs |
| `voice_cloning_and_ethics.py` | The safety angle |
| `voice_agents_architecture.py` | STT → LLM → TTS + interruption |
| `audio_evaluation.py` | WER, MOS, DER, end-to-end |

---

*Previous: [← Document Intelligence](../document_intelligence/README.md) · Next: [Video Understanding →](../video_understanding/README.md)*  ·  *Back to [main README](../../README.md)*

# Audio Pipeline Guide

## The full stack for a voice agent

```
User mic → VAD (Silero) → Streaming ASR (Whisper) → Turn-end detection
        → Full transcript → LLM (agent + tools) → Response text
        → Streaming TTS (XTTS/ElevenLabs) → User speakers
```

## Latency budget for natural conversation

- End-of-user-speech to first-audio-response: **<800ms** natural, <1500ms tolerable.
- VAD end-of-turn: 200-400ms.
- ASR final: 100-200ms.
- LLM first token: 100-300ms.
- TTS first audio: 100-200ms.

## Whisper choices

- Real-time streaming: tiny or base.
- Batch high-quality: large-v3 or turbo.
- Multilingual: large-v3 has best coverage.
- Serving: faster-whisper (CTranslate2) — 4× faster than reference.

## TTS choices

- Open: XTTS v2, Bark, StyleTTS 2, Piper.
- Commercial: ElevenLabs (best quality + cloning), PlayHT, OpenAI TTS.
- Streaming TTS starts audio while generating (sub-500ms first audio).

## Interruption handling

- Detect user speech during TTS playback.
- Halt TTS immediately.
- Buffer partial LLM output as context.
- Resume or restart based on user intent.

## Cost profile

- Whisper self-hosted: ~$0.001-0.01/min.
- Whisper API: $0.006-0.02/min.
- TTS open-source: $0.001-0.01/1000 chars.
- TTS commercial (ElevenLabs): $0.02-0.30/1000 chars.

## Ethics

- Voice cloning: consent, watermarking, detection.
- Legal: EU AI Act, evolving US legislation.

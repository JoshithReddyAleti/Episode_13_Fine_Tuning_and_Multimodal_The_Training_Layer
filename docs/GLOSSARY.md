# Glossary

**AdaLoRA** — LoRA variant with adaptive per-layer rank.

**Adapter** — small trainable module (LoRA, IA³) attached to a frozen base.

**Alignment** — training a model to prefer certain behaviors/outputs.

**AlpacaEval** — LLM-judge benchmark comparing to a reference model.

**Axolotl** — config-driven fine-tuning framework.

**Bradley-Terry** — preference model underlying pairwise loss.

**Catastrophic forgetting** — losing base capabilities during fine-tuning.

**Chat template** — model-specific format for conversations.

**CLIP** — joint image-text embedding model.

**ColPali** — retrieval based on visual page embeddings.

**Constitutional AI** — Anthropic's principle-based alignment approach.

**CPT (Continued Pretraining)** — next-token training on domain corpus.

**DARE** — drop-and-rescale preprocessing for model merging.

**DeepSpeed ZeRO** — Microsoft's sharded training implementation.

**Diarization** — determining who said what in multi-speaker audio.

**Distillation** — training a smaller model to imitate a larger.

**Donut** — OCR-free document understanding model.

**DoRA** — weight-decomposed LoRA.

**DPO** — Direct Preference Optimization.

**Evol-Instruct** — synthetic data method that evolves instructions to be more complex.

**FSDP** — Fully Sharded Data Parallel, PyTorch's ZeRO equivalent.

**Full FT** — full fine-tuning (all params updated).

**IA³** — Infused Adapter by Inhibiting and Amplifying Inner Activations.

**IPO** — Identity Preference Optimization (DPO variant).

**KTO** — Kahneman-Tversky Optimization (binary preferences).

**LayoutLM** — layout-aware document model family.

**LLaVA** — dominant open-source VLM architecture.

**LongLoRA** — LoRA optimized for extending context length.

**LoRA** — Low-Rank Adaptation.

**MergeKit** — the standard model merging tool.

**MMLU** — Massive Multitask Language Understanding benchmark.

**Multi-LoRA** — serving many adapters over one base.

**Mel spectrogram** — time-frequency audio representation.

**NF4** — 4-bit NormalFloat quantization.

**ORPO** — Odds Ratio Preference Optimization.

**PEFT** — Parameter-Efficient Fine-Tuning.

**PPO** — Proximal Policy Optimization.

**Preference optimization** — training on chosen/rejected pairs.

**QLoRA** — LoRA + 4-bit quantization + paged optimizer.

**Reward model (RM)** — model predicting human preference score.

**RLHF** — Reinforcement Learning from Human Feedback.

**SFT** — Supervised Fine-Tuning.

**SigLIP** — improved CLIP with sigmoid loss.

**SimPO** — Simple Preference Optimization (no reference model).

**S-LoRA** — specialized system for high-throughput multi-LoRA serving.

**TIES** — Trim, Elect Sign, and Merge — conflict-aware model merging.

**TTS** — Text-to-Speech.

**Unsloth** — third-party training library, 2× faster than naive HF.

**VAD** — Voice Activity Detection.

**VLM** — Vision-Language Model.

**Whisper** — OpenAI's speech recognition model.

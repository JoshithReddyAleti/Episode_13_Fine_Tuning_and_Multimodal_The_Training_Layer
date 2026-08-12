# Decision Framework — Every Major Choice

## Q: Fine-tune, RAG, or better prompting?

1. Prompt with strongest base. Measure.
2. Not enough? Add few-shot.
3. Not enough because of missing knowledge? RAG.
4. Not enough because of style/format/skill? Fine-tune.

## Q: Full FT or PEFT?

- PEFT (LoRA/QLoRA) is the modern default.
- Full FT only for: continued pretraining, substantial capability transfer, or when quality justifies cost.

## Q: LoRA rank?

- Style: 4-8.
- Instruction: 8-16.
- Complex task: 16-32.
- Approaching full FT: 32-64.
- Start r=8, iterate based on eval.

## Q: LoRA vs QLoRA?

- Model fits FP16/BF16: LoRA.
- Doesn't fit: QLoRA (NF4).
- Very large on limited hardware: QLoRA.

## Q: SFT then DPO, or ORPO?

- Already have SFT: SFT → DPO.
- Building from scratch with preference data: ORPO (single-stage).

## Q: PPO or DPO?

- Default: DPO. Simpler, more stable.
- PPO only with mature RM + team + infrastructure.

## Q: DDP, FSDP, or DeepSpeed?

- Model fits on one GPU + want speed: DDP.
- Doesn't fit + PyTorch-native preference: FSDP.
- Existing DeepSpeed setup or specific features needed: DeepSpeed.

## Q: Merge adapter or serve as LoRA?

- Single-tenant, forever: merge.
- Multi-tenant: keep as LoRA, multi-LoRA serving.
- Iterating: keep as LoRA.

## Q: Text RAG vs ColPali?

- Text-heavy documents, simple layout: text RAG.
- Layout / figures / charts matter: ColPali.
- Both matter: hybrid.

## Q: VLM vs OCR + LLM?

- Cost/scale/simple docs: OCR + LLM.
- Layout-critical / complex: VLM.
- Best-in-class: hybrid (VLM for complex regions).

## Q: Voice agent — open or commercial?

- Prototype / cost-sensitive: open (Whisper + XTTS).
- Best quality + willing to pay: ElevenLabs + strong LLM.
- Enterprise features (telephony, compliance): commercial platforms (LiveKit, Pipecat, Vapi, Retell).

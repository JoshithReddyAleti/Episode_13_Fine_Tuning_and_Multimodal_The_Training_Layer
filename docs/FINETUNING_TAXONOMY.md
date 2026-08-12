# Fine-Tuning Taxonomy

Complete vocabulary map for the fine-tuning discipline.

## By training objective

- **SFT (Supervised Fine-Tuning)** — next-token prediction on (input, output) pairs.
- **CPT (Continued Pretraining)** — next-token prediction on domain corpus.
- **Preference Optimization** — updating the model based on which of two outputs a human prefers.
  - RLHF (PPO) — RL with a reward model.
  - DPO — direct preference optimization (no RL).
  - IPO / KTO / ORPO / SimPO — DPO variants.
- **Reinforcement Learning (task-based)** — RL where rewards come from environment (code execution, math correctness).

## By parameter modification

- **Full FT** — all parameters updated.
- **PEFT (Parameter-Efficient Fine-Tuning)** — small subset updated or added.
  - LoRA / QLoRA / DoRA / LongLoRA / AdaLoRA / IA³.
  - Prompt tuning, prefix tuning.

## By deployment

- **Merged** — LoRA baked into base weights.
- **Adapter-served** — LoRA loaded dynamically.
- **Multi-tenant multi-LoRA** — many adapters, one base.

## By purpose

- **Style / format** — how the model outputs.
- **Task-specific** — narrow capability improvement.
- **Domain adaptation** — knowledge and vocabulary.
- **Alignment** — preferences and refusals.
- **Capability transfer** — teach new skills via distillation.

See section READMEs for depth on each.

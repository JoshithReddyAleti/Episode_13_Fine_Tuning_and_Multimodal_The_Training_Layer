# 🏗️ Training Infrastructure — Making Training Actually Run

> *Config sheets that would make training run "work" on paper often OOM at step 500, or produce NaN at step 5000, or crash at 47% and lose your job. This section is the infrastructure that keeps training runs alive.*

---

## Single-GPU Training (`single_gpu_training.py`)

Starting point for most task-specific fine-tunes.

**What fits on a single GPU (2026):**
- **H100/A100 80GB:** 7B QLoRA comfortable; 13B QLoRA tight; 70B QLoRA feasible with care.
- **L40S 48GB:** 7B QLoRA comfortable; 13B tight.
- **RTX 4090 24GB:** 7B QLoRA short context; 3B full FT.

**Stack:** model in 4-bit (NF4), LoRA adapter in BF16, paged optimizer (8-bit paged Adam), gradient checkpointing, flash attention.

**Frameworks:** **Unsloth** (~2× faster, ~40% less memory than naive HF), **Axolotl** (config-driven), **HF Transformers + PEFT + TRL** (more setup).

**7B QLoRA recipe:**
```yaml
model: meta-llama/Llama-3.1-8B-Instruct
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 2e-4
gradient_checkpointing: true
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
max_seq_length: 2048
```

Trains 4-8 hours on 10K examples.

---

## Multi-GPU Training (`multi_gpu_training.py`)

**Three paradigms:**

**DDP (DistributedDataParallel):** each GPU full copy; data split; gradients synced via AllReduce. Scales ~linear up to interconnect. Limitation: model must fit on one GPU.

**FSDP (Fully Sharded Data Parallel):** parameters, gradients, optimizer states sharded across GPUs. On-the-fly gather/scatter. Scales to very large models. ~10-30% throughput hit vs DDP.

**DeepSpeed ZeRO:** similar idea, Microsoft. Three stages. More mature, more configurable, more complex.

**When each:**
- Model fits comfortably on one GPU, multi-GPU for speed → DDP.
- Doesn't fit → FSDP or DeepSpeed ZeRO.
- Existing DeepSpeed familiarity → DeepSpeed.

---

## DeepSpeed ZeRO (`deepspeed_zero.py`)

**Three stages of progressive sharding:**

**Stage 1:** optimizer states sharded. ~4× memory reduction (Adam states). Simplest.

**Stage 2:** optimizer + gradients sharded. ~8× reduction. Small comm overhead vs Stage 1.

**Stage 3:** everything sharded. Full model rarely resident on one GPU. Maximum reduction. More communication.

**Config example (Stage 3):**
```yaml
zero_optimization:
  stage: 3
  offload_optimizer:
    device: cpu
    pin_memory: true
  offload_param:
    device: cpu
    pin_memory: true
  overlap_comm: true
  contiguous_gradients: true
```

**With CPU offload:** train models larger than aggregate GPU memory, slower.

Framework integration: `transformers` Trainer native support via config file.

---

## FSDP Deep Dive (`fsdp_deep_dive.py`)

**PyTorch's answer to ZeRO Stage 3.**

**How it works:**
1. Model divided into "units" (typically transformer layers).
2. Each unit's params sharded across GPUs.
3. When a layer computed: all-gather from all GPUs → forward/backward → reshard.
4. Gradients reduce-scattered — each GPU keeps only its shard.

**FSDP2 (2024+):** newer implementation with cleaner semantics, better performance. Prefer for new setups.

**Configuration:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={LlamaDecoderLayer}),
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
)
```

**FSDP vs DeepSpeed:** FSDP is PyTorch-native, simpler; DeepSpeed has more features (curriculum, MoE, mature CPU offload). Both scale to 70B+. **FSDP is increasingly the default for new PyTorch pipelines in 2026.**

---

## Accelerate and TRL (`accelerate_and_trl.py`)

**HuggingFace ecosystem for distributed training.**

**`accelerate`:** wraps DDP, FSDP, DeepSpeed. `accelerate config` interactive setup. `accelerate launch train.py` runs across GPUs/nodes.

**`trl`:** higher-level trainers. `SFTTrainer`, `DPOTrainer`, `PPOTrainer`, `ORPOTrainer`, `KTOTrainer`, etc. Integrates with PEFT (LoRA/QLoRA) natively.

**Typical script:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=SFTConfig(output_dir="./out", per_device_train_batch_size=4, num_train_epochs=3),
)
trainer.train()
```

Modern default.

---

## Unsloth Optimizations (`unsloth_optimizations.py`)

Third-party library providing ~2× speedup, ~40% memory reduction over naive HF.

**How:** custom Triton kernels for critical ops (attention, MLP, LoRA). Manual QLoRA forward/backward optimization. Aggressive kernel fusion.

**When:** single/dual GPU, task-specific fine-tuning, 7B-70B models.

**When it doesn't help:** very large-scale multi-node (Unsloth is single-node-focused); custom architectures not yet supported; very small models where kernel launches dominate.

Widely used for cost-effective single-GPU fine-tuning.

---

## Axolotl Framework (`axolotl_framework.py`)

Config-driven fine-tuning framework. Everything in YAML, one command runs.

**Handles:** dataset loading/formatting, chat template application, LoRA/QLoRA config, DDP/FSDP/DeepSpeed via accelerate, sample packing, multiple methods (SFT, DPO, KTO, ORPO), W&B/MLflow.

**Example config:**
```yaml
base_model: meta-llama/Llama-3.1-8B-Instruct
load_in_4bit: true
datasets:
  - path: myorg/my-dataset
    type: alpaca
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_target_linear: true
sequence_len: 2048
sample_packing: true
gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 3
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 2e-4
bf16: true
gradient_checkpointing: true
flash_attention: true
warmup_ratio: 0.03
```

Run: `accelerate launch -m axolotl.cli.train config.yaml`.

Everything in one file. Reproducible. Easy to share.

---

## LlamaFactory (`llama_factory.py`)

Alternative to Axolotl with a UI option. Similar functionality. Popular in Asian AI dev community, UI-first for beginners.

Axolotl vs LlamaFactory: pick whichever your team gravitates to.

---

## Training on Cloud (`training_on_cloud.py`)

**On-demand:** Runpod (cheap, spot options), Lambda Labs (quality, reserved), Modal (serverless), CoreWeave (enterprise-grade), AWS/GCP/Azure (expensive, integrated).

**Managed:** Together AI, Fireworks, HF Hub.

**Self-managed (bare metal):** best economics at scale, needs ops team.

**Typical cost for 7B QLoRA (one iteration):**
- Runpod H100 6 hours: $15-25.
- Managed API (Together): $30-100.
- Bare metal: ~$5 electricity + capital.

**Rule:** iterate on cheap infra → move to enterprise cloud when production requires.

---

## Training Observability (`training_observability.py`)

**Tools:** Weights & Biases (dominant), MLflow (open-source), TensorBoard (OG), Neptune/Comet (commercial).

**What to log:** loss (train/val), LR, gradient norm, throughput (tokens/sec), GPU memory, sample generations at intervals, full config.

**Advanced:** attention visualizations, per-layer stats, preference-based sample rating.

**Setup discipline:** every run gets unique ID. Full config logged. Sample outputs archived. If run turns out great in 3 months, you can find and reproduce.

---

## Recovering From Training Crashes (`recovering_from_training_crashes.py`)

Long training runs *will* crash. Not "might."

**Common crashes:** OOM at longer-than-usual batch; NCCL timeout; node preempted (spot); storage failure; silent hang (deadlock).

**Preparation:**

**Checkpointing:** save every N steps (chosen so <30 min lost). Full checkpoint (model, optimizer, LR scheduler, RNG) for exact resume.

**Storage:** persistent volumes. Regular sync to remote. Multiple recent checkpoints — corruption of latest possible.

**Restart logic:** trainer autodetects and resumes. Verify dataset ordering / random state preserved.

**Monitoring:** alert if training halts (no metric in 10 minutes). On-call rotation at critical stages.

**Debugging:** save full traceback and last N steps. Reproduce with subset. Check hardware (`nvidia-smi` errors, dmesg). Sometimes just retry — transient failures happen.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `single_gpu_training.py` | QLoRA on one GPU |
| `multi_gpu_training.py` | DDP, FSDP, DeepSpeed overview |
| `deepspeed_zero.py` | ZeRO stages 1, 2, 3 |
| `fsdp_deep_dive.py` | PyTorch-native sharding |
| `accelerate_and_trl.py` | HuggingFace ecosystem |
| `unsloth_optimizations.py` | Speed + memory |
| `axolotl_framework.py` | Config-driven training |
| `llama_factory.py` | Alternative framework |
| `training_on_cloud.py` | Where to run |
| `training_observability.py` | W&B, MLflow |
| `recovering_from_training_crashes.py` | Checkpoint discipline |

---

*Previous: [← Continued Pretraining](../continued_pretraining/README.md) · Next: [Fine-Tuning Evaluation →](../finetuning_evaluation/README.md)*  ·  *Back to [main README](../../README.md)*

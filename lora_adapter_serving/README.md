# 🎛️ LoRA Adapter Serving — Multi-Tenant Fine-Tunes At Scale

> *One base model, N tenant-specific adapters. Each adapter is 10-100 MB. Serving 100 tenants for the memory cost of 1 base model. This is the economics that make custom fine-tuning viable for enterprise.*

---

## Adapter Merging (`adapter_merging.py`)

**Merging** = bake the LoRA update into base weights permanently.

**Math:** `W_final = W_base + α · (B · A) / r`

**When to merge:**
- **Single-tenant** — one fine-tune forever.
- **Distilling to compressed format** — quantize (AWQ, GPTQ, GGUF). LoRA adapters can't easily.
- **No adapter-swap requirement.**

**When not to merge:**
- Multi-tenant.
- Iterating (may retrain often).
- Composition (combine adapters at inference).

**Practical:**
```python
from peft import PeftModel
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
peft_model = PeftModel.from_pretrained(model, "path/to/adapter")
merged = peft_model.merge_and_unload()
merged.save_pretrained("path/to/merged")
```

**Storage:** merged = base size. Adapter = ~50 MB. Merging is a huge storage multiplier with many adapters.

---

## Dynamic Adapter Loading (`dynamic_adapter_loading.py`)

**Load at request time, apply, respond, potentially unload.**

**Pattern:**
1. Base resident in HBM.
2. Adapter loaded lazily on first request per tenant.
3. Cached in HBM for subsequent.
4. LRU eviction if HBM cache fills.

**Trade-offs:**
- **First-request latency:** disk/S3 load adds 100ms-2s.
- **Steady-state:** cache hit → sub-ms overhead.
- **Memory:** each cached adapter uses HBM proportional to params.

**Frameworks:** vLLM (`--enable-lora --max-loras N`), TGI, SGLang. HF Transformers + PEFT slower than production servers.

**Production:** adapter registry (below), health check on adapter loading (corrupt adapters), metrics (load latency, cache hit rate, eviction rate).

---

## Multi-LoRA Inference (`multi_lora_inference.py`)

**Serving different adapters concurrently in the same batch.** What makes multi-tenant efficient.

**Naive:** 8 requests, 8 adapters = 8 separate forward passes.

**Efficient (S-LoRA/Punica style):** base forward once for whole batch, then per-request adapters via batched matmul.

**vLLM's multi-LoRA:**
- `--enable-lora --max-loras 8` at server start.
- Requests specify adapter via `model` field.
- Batched natively; near-zero overhead vs base alone.

**Performance:** throughput ~90-95% of pure base. Concurrent capacity limited by adapter cache size, not adapter count. Latency: negligible overhead.

**Constraints:** all adapters same rank (framework limitation); same target modules; adapter cache in HBM.

---

## S-LoRA and Punica (`slora_and_punica.py`)

**Specialized systems for very-high-scale multi-LoRA.**

**S-LoRA (Stanford 2023):** thousands of adapters. Unified paging for KV cache and adapter cache. Custom CUDA kernels for batched LoRA matmul. ~4× throughput improvement over vLLM on adapter-heavy workloads.

**Punica (Chen et al. 2024):** similar goals, different implementation. Segmented gather kernels.

**When:** >50 concurrent unique adapters; adapter loading is real bottleneck; adapter cache memory is constraint.

**In practice (2026):** most teams use vLLM's built-in. Specialized systems for very large deployments (thousands of adapters, tens of thousands QPS).

---

## Adapter Composition (`adapter_composition.py`)

**Combining adapters at inference for combined behavior.**

**Weighted sum:** `W_effective = W_base + λ_A · A_delta + λ_B · B_delta`. Framework via `peft.add_weighted_adapter`.

**Task arithmetic:** adapters "added" and "subtracted."

**DARE-style:** randomly drop most updates before summing. Reduces conflicts.

**Where this shines:** customer voice + task adapters; domain adapters (medical + Spanish); user-configurable style/task blending.

**Caveats:** doesn't always produce clean combined behavior; some pairs conflict; test compositions you'll serve.

---

## Adapter Registry (`adapter_registry.py`)

**Source of truth for what adapters exist and how configured.**

**Registry entry:** adapter ID/name, base model reference, version, storage location (S3, HF Hub, local), training metadata (dataset, hyperparameters, run), evaluation metrics, lifecycle stage (dev/staging/prod/deprecated), owner, access control.

**Operations:** register, list (filter by tenant/base/stage), get metadata, deploy (link with rollout%), deprecate.

**Tools:** HuggingFace private hub (storage), MLflow model registry (metadata + versioning), W&B artifacts (training-integrated), custom internal.

**Integration:** inference server periodically syncs adapter list from registry.

---

## Per-Tenant Adapters (`per_tenant_adapters.py`)

**The multi-tenant pattern.**

**Architecture:**
```
Application → Gateway (looks up tenant → adapter mapping)
                ↓
              vLLM (base + adapter cache)
                ↓
              Response
```

Base chosen for tier (Llama-3.1-8B or 70B). Adapters per-tenant on their data. Gateway maps `X-Tenant-Id` header → adapter name.

**Economics of N tenants:**
- **N separate full fine-tunes:** N × model size. N servers. Cost = N × single-model.
- **1 base + N adapters:** 1 model + N × ~50 MB in HBM. 1-few servers.
- **Cost per tenant at N=100:** 20-50× lower than dedicated.

**Constraints:** adapters from same base; same rank/target modules; cache limits concurrent unique adapters.

**Real-world:** enterprise platform serving 500 client orgs, each fine-tuned. Ops burden is per-adapter management, not per-model.

---

## Adapter Serving Economics (`adapter_serving_economics.py`)

**Per-adapter costs:**
- Training: $30-200 QLoRA.
- Storage: ~$0.001/month (S3).
- Evaluation: $10-100.
- Retraining: $30-200 (quarterly typical).

**Per-request costs:**
- Base amortized across tenants.
- Adapter cache load amortized across requests.
- Marginal per adapter-swap: ~$0.00001.

**Per tenant at scale:**
- 50 tenants: $0.10-1/request avg.
- 500 tenants on 1 base: $0.02-0.20.
- 5000 tenants across 5 tiers: $0.01-0.05.

**Cross-subsidization:** heavy tenants can subsidize light ones since base cost is fixed. Pricing matters.

**Fair sharing:** ensure noisy neighbors don't degrade others. Rate limits, priority tiers, dedicated queues.

**When multi-LoRA doesn't beat dedicated:** one tenant >20% of load; different base needed; radically different quantization/serving config.

---

## Files in This Directory

| File | What It Covers |
|---|---|
| `adapter_merging.py` | When and how to merge |
| `dynamic_adapter_loading.py` | Lazy loading + caching |
| `multi_lora_inference.py` | Batched multi-LoRA serving |
| `slora_and_punica.py` | High-throughput specialized |
| `adapter_composition.py` | Combining adapters at inference |
| `adapter_registry.py` | Source of truth |
| `per_tenant_adapters.py` | Multi-tenant pattern |
| `adapter_serving_economics.py` | Cost model |

---

*Previous: [← Fine-Tuning Evaluation](../finetuning_evaluation/README.md) · Next: [Model Merging →](../model_merging/README.md)*  ·  *Back to [main README](../../README.md)*

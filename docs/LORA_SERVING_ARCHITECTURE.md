# LoRA Serving Architecture — Multi-Tenant at Scale

## The pattern

One base model. N per-tenant adapters. Served through vLLM multi-LoRA.

## Reference architecture

```
Application → Gateway (auth, tenant → adapter mapping)
              ↓
            vLLM Server (base + adapter cache in HBM)
              ↓
              Response
```

## Adapter Registry

Source of truth. Every entry:
- adapter ID, base model reference, version.
- Storage location (S3 URI, HF Hub).
- Training metadata (dataset, hyperparameters, run).
- Evaluation metrics at time of registration.
- Lifecycle stage (dev/staging/prod/deprecated).
- Owner.

Operations: register, list, get, deploy (with rollout %), deprecate.

## Serving configuration (vLLM)

```
--enable-lora
--max-loras 16               # concurrent unique adapters in memory
--max-lora-rank 32           # max rank supported
--lora-modules a1=path b1=path ...
```

## Cost profile at N tenants

- N=50: $0.10-1 per request average.
- N=500 on 1 base: $0.02-0.20.
- N=5000 across 5 tiers: $0.01-0.05.

vs dedicated fine-tunes: 20-50× cost reduction.

## Constraints

- All adapters same rank (framework limitation).
- All adapters same target modules.
- HBM adapter cache size limits concurrent unique adapters per server.

## When multi-LoRA doesn't beat dedicated

- One tenant using >20% of aggregate load.
- Tenant needs different base model.
- Radically different quantization/serving config per tenant.

"""
Training utilities: cost/time estimators, LoRA sizing, LR range helper.

Usage:
    python -m src.utils.training_utils cost --model-b 8 --n-tokens 5e7 --gpu h100
    python -m src.utils.training_utils lora-size --model-b 8 --rank 16
    python -m src.utils.training_utils lr-range --model-b 8 --peft qlora
"""

import argparse
import json
import sys
from typing import Any


# Approximate H100/A100 throughput in tokens/sec for training
GPU_THROUGHPUT_TRAINING = {
    "h100": {"qlora_7b": 8000, "qlora_13b": 5000, "qlora_70b": 900, "full_7b": 3000, "full_70b": 200},
    "a100": {"qlora_7b": 4500, "qlora_13b": 2800, "qlora_70b": 500, "full_7b": 1600, "full_70b": 110},
    "l40s": {"qlora_7b": 3500, "qlora_13b": 2000, "qlora_70b": 250, "full_7b": 1200, "full_70b": 60},
    "4090": {"qlora_7b": 1800, "qlora_13b": 1000, "qlora_70b": 0, "full_7b": 400, "full_70b": 0},
}

# Approximate on-demand hourly cost (USD)
GPU_HOURLY = {"h100": 3.50, "a100": 2.20, "l40s": 1.30, "4090": 0.50}


def estimate_cost(model_b: float, n_tokens: float, gpu: str = "h100", method: str = "qlora") -> dict[str, Any]:
    """Estimate training cost/time."""
    model_key = f"{method}_{'7b' if model_b <= 8 else '13b' if model_b <= 15 else '70b'}"
    tps = GPU_THROUGHPUT_TRAINING.get(gpu, {}).get(model_key, 0)
    if not tps:
        return {"error": f"Unsupported combo: {gpu} / {method} / {model_b}B"}
    hours = n_tokens / tps / 3600
    cost = hours * GPU_HOURLY.get(gpu, 0)
    return {
        "model_b": model_b,
        "n_tokens": n_tokens,
        "gpu": gpu,
        "method": method,
        "throughput_tokens_per_sec": tps,
        "estimated_hours": round(hours, 2),
        "estimated_cost_usd": round(cost, 2),
    }


def lora_size(model_b: float, rank: int = 16, target: str = "attn_mlp", hidden: int = 4096) -> dict[str, Any]:
    """Estimate LoRA adapter size."""
    # Rough model hidden size for common sizes
    hidden_by_size = {7: 4096, 8: 4096, 13: 5120, 70: 8192}
    h = hidden_by_size.get(int(round(model_b)), hidden)

    # Number of layers heuristic
    n_layers = {7: 32, 8: 32, 13: 40, 70: 80}.get(int(round(model_b)), 32)

    # Modules per layer
    if target == "attn_only":
        modules = 2  # q, v
    elif target == "attn_full":
        modules = 4  # q, k, v, o
    else:  # attn_mlp
        modules = 7  # q, k, v, o, gate, up, down

    # LoRA params per module: rank * (in_dim + out_dim). Simplified to 2 * rank * h.
    params_per_module = 2 * rank * h
    total_params = n_layers * modules * params_per_module
    size_mb_bf16 = total_params * 2 / (1024 * 1024)  # 2 bytes per param in bf16

    return {
        "model_b": model_b,
        "rank": rank,
        "target": target,
        "hidden": h,
        "n_layers": n_layers,
        "modules_per_layer": modules,
        "total_trainable_params": total_params,
        "adapter_size_mb_bf16": round(size_mb_bf16, 2),
    }


def lr_range(model_b: float, peft: str = "qlora") -> dict[str, Any]:
    """Suggest LR range for a given model size and PEFT method."""
    ranges = {
        "full": (1e-6, 1e-5),
        "lora": (1e-5, 1e-4),
        "qlora": (1e-4, 5e-4),
        "prompt_tuning": (1e-3, 1e-2),
    }
    low, high = ranges.get(peft, (1e-5, 1e-4))
    # Very rough size adjustment
    if model_b >= 30:
        low *= 0.5
        high *= 0.5
    return {
        "model_b": model_b,
        "peft": peft,
        "suggested_lr_low": low,
        "suggested_lr_high": high,
        "recommended_start": (low + high) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Training utilities for fine-tuning")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cost = sub.add_parser("cost", help="Estimate training cost/time")
    p_cost.add_argument("--model-b", type=float, required=True)
    p_cost.add_argument("--n-tokens", type=float, required=True)
    p_cost.add_argument("--gpu", default="h100")
    p_cost.add_argument("--method", default="qlora", choices=["qlora", "full"])

    p_lora = sub.add_parser("lora-size", help="Estimate LoRA adapter size")
    p_lora.add_argument("--model-b", type=float, required=True)
    p_lora.add_argument("--rank", type=int, default=16)
    p_lora.add_argument("--target", default="attn_mlp")

    p_lr = sub.add_parser("lr-range", help="Suggest LR range")
    p_lr.add_argument("--model-b", type=float, required=True)
    p_lr.add_argument("--peft", default="qlora")

    args = parser.parse_args()

    if args.cmd == "cost":
        result = estimate_cost(args.model_b, args.n_tokens, args.gpu, args.method)
    elif args.cmd == "lora-size":
        result = lora_size(args.model_b, args.rank, args.target)
    elif args.cmd == "lr-range":
        result = lr_range(args.model_b, args.peft)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Multimodal utilities: image token estimation, cost estimation for VLMs and multimodal APIs.

Usage:
    python -m src.utils.multimodal_utils img-tokens --width 1024 --height 1024 --patch 14
    python -m src.utils.multimodal_utils vlm-cost --n-images 1000 --avg-tokens 800 --model gpt4o
    python -m src.utils.multimodal_utils video-tokens --duration-sec 60 --fps 1 --tokens-per-frame 576
"""

import argparse
import json
import sys
from typing import Any


# Approximate per-1M-token pricing for multimodal-capable models (USD)
MODEL_PRICING_PER_1M = {
    "gpt4o": {"input": 2.50, "output": 10.00},
    "claude": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 1.25, "output": 5.00},
    "qwen_vl": {"input": 0.20, "output": 0.60},  # rough self-hosted
    "llama_vision": {"input": 0.15, "output": 0.50},
}


def img_tokens(width: int, height: int, patch: int = 14) -> dict[str, Any]:
    """Estimate patch tokens for a VLM given image size and patch size."""
    patches_w = width // patch
    patches_h = height // patch
    total_patches = patches_w * patches_h
    return {
        "width": width,
        "height": height,
        "patch": patch,
        "patches_w": patches_w,
        "patches_h": patches_h,
        "total_patch_tokens": total_patches,
    }


def vlm_cost(n_images: int, avg_tokens_per_image: int, model: str = "gpt4o",
             text_tokens_in: int = 100, output_tokens: int = 300) -> dict[str, Any]:
    """Estimate cost for N VLM queries with images."""
    pricing = MODEL_PRICING_PER_1M.get(model)
    if not pricing:
        return {"error": f"Unknown model: {model}. Options: {list(MODEL_PRICING_PER_1M.keys())}"}

    total_input = n_images * (avg_tokens_per_image + text_tokens_in)
    total_output = n_images * output_tokens

    cost_in = total_input * pricing["input"] / 1_000_000
    cost_out = total_output * pricing["output"] / 1_000_000

    return {
        "n_images": n_images,
        "avg_tokens_per_image": avg_tokens_per_image,
        "model": model,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "cost_input_usd": round(cost_in, 2),
        "cost_output_usd": round(cost_out, 2),
        "total_cost_usd": round(cost_in + cost_out, 2),
        "cost_per_image_usd": round((cost_in + cost_out) / n_images, 4),
    }


def video_tokens(duration_sec: float, fps: float = 1.0, tokens_per_frame: int = 576) -> dict[str, Any]:
    """Estimate total tokens for a video at given fps + tokens per frame."""
    n_frames = duration_sec * fps
    total_tokens = int(n_frames * tokens_per_frame)
    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "tokens_per_frame": tokens_per_frame,
        "n_frames": n_frames,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal utilities: token & cost estimation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("img-tokens", help="Estimate patch tokens for an image")
    p_img.add_argument("--width", type=int, required=True)
    p_img.add_argument("--height", type=int, required=True)
    p_img.add_argument("--patch", type=int, default=14)

    p_vlm = sub.add_parser("vlm-cost", help="Estimate cost for N VLM queries")
    p_vlm.add_argument("--n-images", type=int, required=True)
    p_vlm.add_argument("--avg-tokens", type=int, required=True)
    p_vlm.add_argument("--model", default="gpt4o")
    p_vlm.add_argument("--text-tokens-in", type=int, default=100)
    p_vlm.add_argument("--output-tokens", type=int, default=300)

    p_vid = sub.add_parser("video-tokens", help="Estimate total tokens for a video")
    p_vid.add_argument("--duration-sec", type=float, required=True)
    p_vid.add_argument("--fps", type=float, default=1.0)
    p_vid.add_argument("--tokens-per-frame", type=int, default=576)

    args = parser.parse_args()

    if args.cmd == "img-tokens":
        result = img_tokens(args.width, args.height, args.patch)
    elif args.cmd == "vlm-cost":
        result = vlm_cost(args.n_images, args.avg_tokens, args.model,
                          args.text_tokens_in, args.output_tokens)
    elif args.cmd == "video-tokens":
        result = video_tokens(args.duration_sec, args.fps, args.tokens_per_frame)
    else:
        parser.print_help()
        sys.exit(1)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

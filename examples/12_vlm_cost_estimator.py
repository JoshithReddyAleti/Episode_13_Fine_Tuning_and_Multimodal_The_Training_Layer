'''Estimate VLM inference cost for 1000 images across models.'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import multimodal_utils

if __name__ == "__main__":
    for model in ["gpt4o", "claude", "gemini", "qwen_vl", "llama_vision"]:
        r = multimodal_utils.vlm_cost(1000, 800, model)
        if "total_cost_usd" in r:
            print(f"{model}: ${r['total_cost_usd']:.2f} for 1000 images "
                  f"(${r['cost_per_image_usd']:.4f}/image)")

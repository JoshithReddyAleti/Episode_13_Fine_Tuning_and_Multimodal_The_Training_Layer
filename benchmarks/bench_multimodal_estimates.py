'''Benchmark multimodal cost/token estimate throughput.'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.utils import multimodal_utils


def main():
    t0 = time.perf_counter()
    for w in [224, 336, 448, 672, 1024]:
        for h in [224, 336, 448, 672, 1024]:
            for p in [14, 16]:
                multimodal_utils.img_tokens(w, h, p)
    dt = time.perf_counter() - t0
    print(f"25 img_token configs: {dt*1000:.2f}ms")

    t0 = time.perf_counter()
    for model in ["gpt4o", "claude", "gemini", "qwen_vl", "llama_vision"]:
        multimodal_utils.vlm_cost(1000, 800, model)
    dt = time.perf_counter() - t0
    print(f"5 vlm_cost configs: {dt*1000:.2f}ms")


if __name__ == "__main__":
    main()


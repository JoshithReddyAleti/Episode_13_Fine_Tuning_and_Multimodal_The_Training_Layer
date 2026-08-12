'''Estimate LoRA adapter size for common configurations.'''

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import training_utils

if __name__ == "__main__":
    for model_b in [7, 13, 70]:
        for rank in [8, 16, 32]:
            result = training_utils.lora_size(model_b, rank=rank, target="attn_mlp")
            print(f"Model {model_b}B, rank {rank}: {result['adapter_size_mb_bf16']} MB adapter, "
                  f"{result['total_trainable_params']:,} trainable params")

'''Suggest LR ranges for common fine-tuning setups.'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import training_utils

if __name__ == "__main__":
    for model_b in [7, 13, 70]:
        for peft in ["qlora", "lora", "full"]:
            r = training_utils.lr_range(model_b, peft)
            print(f"{model_b}B / {peft}: {r['suggested_lr_low']:.0e} to {r['suggested_lr_high']:.0e}")

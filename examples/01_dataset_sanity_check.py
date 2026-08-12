'''Sanity check a training dataset before fine-tuning.'''

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import dataset_utils

if __name__ == "__main__":
    # Create a tiny sample dataset for illustration
    sample = [
        {"messages": [{"role": "user", "content": f"Question {i}"},
                      {"role": "assistant", "content": f"Answer {i}"}]}
        for i in range(50)
    ]
    p = Path("/tmp/sample_dataset.jsonl")
    with p.open("w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    report = dataset_utils.validate_dataset(str(p))
    print(json.dumps(report, indent=2))

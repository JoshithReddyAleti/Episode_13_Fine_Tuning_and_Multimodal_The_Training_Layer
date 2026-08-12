'''Show the structure of preference data for DPO training.'''

import json

if __name__ == "__main__":
    pair = {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris.",
        "rejected": "France's capital is Berlin.",  # incorrect
    }
    print(json.dumps(pair, indent=2))
    print("\nDPO loss pushes P(chosen) up and P(rejected) down relative to reference model.")

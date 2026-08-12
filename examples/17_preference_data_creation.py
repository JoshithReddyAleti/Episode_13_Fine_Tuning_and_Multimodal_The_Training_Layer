'''Illustrate creating preference pairs via UltraFeedback pattern.'''

import json

PROMPT = "Explain quantum entanglement in one sentence."

# Responses from N different models
RESPONSES = {
    "model_A": "Two particles become correlated so measuring one instantly affects the other regardless of distance.",
    "model_B": "Quantum entanglement is when particles are linked and share properties instantaneously.",
    "model_C": "Entanglement means particles can affect each other from far away.",
    "model_D": "Sciencey thing where particles talk.",  # Low quality
}

# Simulated judge scores (0-10)
SCORES = {"model_A": 9, "model_B": 8, "model_C": 6, "model_D": 2}

if __name__ == "__main__":
    ranked = sorted(RESPONSES.keys(), key=lambda k: -SCORES[k])
    pair = {
        "prompt": PROMPT,
        "chosen": RESPONSES[ranked[0]],
        "rejected": RESPONSES[ranked[-1]],
    }
    print(json.dumps(pair, indent=2))

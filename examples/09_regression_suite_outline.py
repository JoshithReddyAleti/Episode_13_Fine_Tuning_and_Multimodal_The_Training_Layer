'''Show what a fine-tuning regression suite covers.'''

REGRESSION_SUITE = {
    "general_knowledge": {"benchmark": "MMLU (subset)", "n": 200, "threshold_drop": 0.05},
    "commonsense": {"benchmark": "HellaSwag (subset)", "n": 200, "threshold_drop": 0.05},
    "reasoning": {"benchmark": "ARC (subset)", "n": 200, "threshold_drop": 0.05},
    "math": {"benchmark": "GSM8K (subset)", "n": 100, "threshold_drop": 0.05},
    "code": {"benchmark": "HumanEval (subset)", "n": 100, "threshold_drop": 0.05},
    "instruction_following": {"benchmark": "IFEval (subset)", "n": 100, "threshold_drop": 0.05},
    "safety": {"benchmark": "custom refusal set", "n": 100, "threshold_drop": 0.10},
}

if __name__ == "__main__":
    print("Regression suite:")
    for capability, spec in REGRESSION_SUITE.items():
        print(f"  {capability}: {spec['benchmark']} ({spec['n']} examples, "
              f"alert if drops >{spec['threshold_drop']*100:.0f}%)")
    print("\nRun every checkpoint before shipping. Any red flag → stop, investigate.")

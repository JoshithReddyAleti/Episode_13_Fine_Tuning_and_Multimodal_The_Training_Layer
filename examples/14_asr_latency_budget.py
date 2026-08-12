'''Show a voice agent latency budget breakdown.'''

BUDGET_MS = {
    "vad_end_of_turn": (200, 400),
    "asr_final": (100, 200),
    "llm_first_token": (100, 300),
    "tts_first_audio": (100, 200),
}

if __name__ == "__main__":
    total_low = sum(l for l, _ in BUDGET_MS.values())
    total_high = sum(h for _, h in BUDGET_MS.values())
    print(f"Total budget: {total_low}-{total_high}ms")
    for stage, (l, h) in BUDGET_MS.items():
        print(f"  {stage:30}: {l}-{h}ms")
    print(f"\nTarget: sub-800ms natural, sub-1500ms tolerable.")

#!/usr/bin/env bash
# Illustrative orchestrator for voice_agent_full_stack.
# Real execution requires a GPU environment and installed deps.
set -euo pipefail

echo "== Voice Agent Full Stack (STT + LLM + TTS) =="
echo "This recipe is documented in README.md. Concrete execution:"
echo "  1. Populate data/train.jsonl."
echo "  2. Adjust config.yaml as needed."
echo "  3. accelerate launch -m axolotl.cli.train config.yaml   # or your preferred framework"

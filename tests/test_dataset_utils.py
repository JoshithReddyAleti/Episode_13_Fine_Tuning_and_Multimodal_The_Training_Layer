import json
import tempfile
from pathlib import Path
from src.utils import dataset_utils


def _write_sample(records: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_validate_dataset_basic():
    records = [
        {"messages": [{"role": "user", "content": f"Q{i}"},
                      {"role": "assistant", "content": f"A{i}"}]}
        for i in range(20)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name
    report = dataset_utils.validate_dataset(path)
    assert report["total_examples"] == 20
    assert report["exact_duplicates"] == 0
    assert "input_length_chars" in report


def test_estimate_tokens():
    records = [{"instruction": "hello", "input": "", "output": "world"}] * 10
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name
    report = dataset_utils.estimate_tokens(path)
    assert report["total_examples"] == 10
    assert report["estimated_tokens"] > 0


def test_detect_duplicates():
    same = {"instruction": "same", "input": "", "output": "same"}
    records = [same, same, same]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name
    report = dataset_utils.validate_dataset(path)
    assert report["exact_duplicates"] > 0


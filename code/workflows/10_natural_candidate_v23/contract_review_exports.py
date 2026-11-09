"""Read the single final contract-label export, filtered by benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def review_label_path(review_dir: Path, reviewer: str) -> Path:
    return review_dir.parent.parent / f"contract_annotator_{reviewer}_labels.jsonl"


def read_review_labels(
    review_dir: Path, reviewer: str, dataset: str
) -> list[dict[str, Any]]:
    with review_label_path(review_dir, reviewer).open(encoding="utf-8") as handle:
        return [
            row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
            if row["dataset"] == dataset
        ]

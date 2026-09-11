from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from data_utils import load_pages, split_pages


def get_split(data_file: str | Path, split: str) -> List[dict]:
    pages = load_pages(data_file)
    train, val, test = split_pages(pages)
    return {"train": train, "val": val, "test": test}[split]


def eval_predictions(predictions: List[List[int]], pages: List[dict]) -> Dict:
    correct_splits = total_splits = 0
    page_perfect = 0
    by_nm = defaultdict(lambda: {"correct": 0, "total": 0, "pages": 0, "perfect": 0})

    for pred, page in zip(predictions, pages):
        gold = [int(x) for x in page["gold_splits"]]
        n, m = int(page["N"]), int(page["M"])
        correct = sum(int(p == g) for p, g in zip(pred, gold))
        correct_splits += correct
        total_splits += len(gold)
        perfect = int(correct == len(gold))
        page_perfect += perfect

        key = "N=M" if n == m else ("N<M" if n < m else "N>M")
        by_nm[key]["correct"] += correct
        by_nm[key]["total"] += len(gold)
        by_nm[key]["pages"] += 1
        by_nm[key]["perfect"] += perfect

    return {
        "split_acc": correct_splits / max(total_splits, 1),
        "page_perfect_rate": page_perfect / max(len(pages), 1),
        "n_pages": len(pages),
        "by_NM": dict(by_nm),
    }


def save_results(name: str, results: Dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")

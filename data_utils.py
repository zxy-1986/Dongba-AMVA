from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_PAGE_KEYS = {
    "N",
    "M",
    "image_files",
    "candidate_fragments",
    "gold_splits",
    "split",
}


def load_pages(data_file: str | Path) -> List[Dict]:
    """Load the processed page-level dataset used by AMVA.

    Expected format: a JSON list. Each page must contain at least:
      - N: number of Dongba image blocks
      - M: number of atomic translated text fragments
      - image_files: filenames relative to data/images
      - candidate_fragments: list[str]
      - gold_splits: N-1 split positions in [1, M-1]
      - split: one of {train, val, test}
    """
    data_file = Path(data_file)
    with data_file.open("r", encoding="utf-8") as f:
        pages = json.load(f)

    if not isinstance(pages, list):
        raise ValueError(f"{data_file} must contain a JSON list of pages.")

    for idx, page in enumerate(pages):
        missing = REQUIRED_PAGE_KEYS.difference(page)
        if missing:
            raise ValueError(f"Page #{idx} is missing keys: {sorted(missing)}")

        n = int(page["N"])
        m = int(page["M"])
        if n < 1 or m < 1:
            raise ValueError(f"Page #{idx}: invalid N={n}, M={m}")
        if len(page["image_files"]) != n:
            raise ValueError(
                f"Page #{idx}: len(image_files)={len(page['image_files'])} != N={n}"
            )
        if len(page["candidate_fragments"]) != m:
            raise ValueError(
                f"Page #{idx}: len(candidate_fragments)={len(page['candidate_fragments'])} != M={m}"
            )
        if len(page["gold_splits"]) != max(n - 1, 0):
            raise ValueError(
                f"Page #{idx}: len(gold_splits)={len(page['gold_splits'])} != N-1={n-1}"
            )
        if page["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Page #{idx}: invalid split={page['split']!r}")

    return pages


def split_pages(pages: Iterable[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train, val, test = [], [], []
    for page in pages:
        split = page["split"]
        if split == "train":
            train.append(page)
        elif split == "val":
            val.append(page)
        elif split == "test":
            test.append(page)
    return train, val, test


def dataset_summary(pages: Iterable[Dict]) -> Dict:
    pages = list(pages)
    train, val, test = split_pages(pages)
    n_values = [int(p["N"]) for p in pages]
    m_values = [int(p["M"]) for p in pages]
    n_lt_m = sum(int(p["N"]) < int(p["M"]) for p in pages)
    return {
        "pages": len(pages),
        "train_pages": len(train),
        "val_pages": len(val),
        "test_pages": len(test),
        "N_min": min(n_values) if n_values else None,
        "N_max": max(n_values) if n_values else None,
        "N_mean": sum(n_values) / len(n_values) if n_values else None,
        "M_min": min(m_values) if m_values else None,
        "M_max": max(m_values) if m_values else None,
        "M_mean": sum(m_values) / len(m_values) if m_values else None,
        "N_lt_M_pages": n_lt_m,
        "N_lt_M_ratio": n_lt_m / len(pages) if pages else None,
    }

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from transformers import ChineseCLIPProcessor

from amva import (
    AMVADataset,
    DEFAULT_CNCLIP_PATH,
    DEFAULT_DATA_FILE,
    DEFAULT_IMAGE_DIR,
    evaluate,
    load_trained_model,
    set_seed,
)
from data_utils import load_pages, split_pages


def _stable_seed(filename: str, base_seed: int) -> int:
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    return base_seed + int(digest, 16)


def make_transform(kind: str, seed: int = 42):
    kind = kind.lower()

    def transform(image: Image.Image, filename: str) -> Image.Image:
        if kind == "clean":
            return image
        if kind == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=2.0))
        if kind == "fade":
            white = Image.new("RGB", image.size, (255, 255, 255))
            return Image.blend(image, white, alpha=0.40)
        if kind == "damage":
            rng = random.Random(_stable_seed(filename, seed))
            out = image.copy()
            w, h = out.size
            target_area = max(1, int(0.06 * w * h))
            # One deterministic white erasure rectangle with approximately 6% area.
            aspect = rng.uniform(0.5, 2.0)
            rw = max(1, min(w, int(math.sqrt(target_area * aspect))))
            rh = max(1, min(h, int(target_area / max(rw, 1))))
            x0 = rng.randint(0, max(w - rw, 0))
            y0 = rng.randint(0, max(h - rh, 0))
            patch = Image.new("RGB", (rw, rh), (255, 255, 255))
            out.paste(patch, (x0, y0))
            return out
        raise ValueError(f"Unknown corruption: {kind}")

    return transform


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for E7 AMVA.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp", default="E7_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--cnclip_path", default=str(DEFAULT_CNCLIP_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="results/robustness_reproduced.json")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_trained_model(
        args.exp, args.checkpoint, args.cnclip_path, device, seed=args.seed
    )
    processor = ChineseCLIPProcessor.from_pretrained(args.cnclip_path)
    _, _, test_pages = split_pages(load_pages(args.data_file))

    results = {}
    for condition in ["clean", "blur", "fade", "damage"]:
        dataset = AMVADataset(
            test_pages,
            processor,
            args.image_dir,
            image_transform=make_transform(condition, seed=args.seed),
        )
        metrics = evaluate(model, dataset, device)
        nm = metrics.get("by_NM", {}).get("N<M", {})
        nm_acc = nm.get("correct", 0) / max(nm.get("total", 1), 1)
        results[condition] = {
            "split_acc": metrics["split_acc"],
            "page_em": metrics["page_perfect_rate"],
            "n_lt_m_acc": nm_acc,
        }
        print(condition, json.dumps(results[condition], ensure_ascii=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained AMVA checkpoint.")
    parser.add_argument("--exp", default="E7_full")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--cnclip_path", default=str(DEFAULT_CNCLIP_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    model = load_trained_model(
        args.exp, args.checkpoint, args.cnclip_path, device, seed=args.seed
    )
    processor = ChineseCLIPProcessor.from_pretrained(args.cnclip_path)

    pages = load_pages(args.data_file)
    train_pages, val_pages, test_pages = split_pages(pages)
    split_map = {"train": train_pages, "val": val_pages, "test": test_pages}
    dataset = AMVADataset(split_map[args.split], processor, args.image_dir)
    metrics = evaluate(model, dataset, device)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

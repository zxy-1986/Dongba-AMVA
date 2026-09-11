from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from baselines.common import eval_predictions, get_split, save_results


def uniform_splits(n: int, m: int):
    return [round(k * m / n) for k in range(1, n)]


def random_splits(n: int, m: int, rng: random.Random):
    if n <= 1:
        return []
    if m < n:
        # This dataset is primarily N<=M; fall back to monotonic uniform positions.
        return uniform_splits(n, m)
    return sorted(rng.sample(range(1, m), n - 1))


def main():
    parser = argparse.ArgumentParser(description="Random/Uniform split baselines.")
    parser.add_argument("--method", choices=["random", "uniform"], required=True)
    parser.add_argument("--data_file", default="data/pages.json")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/baselines")
    args = parser.parse_args()

    pages = get_split(args.data_file, args.split)
    rng = random.Random(args.seed)
    preds = []
    for page in pages:
        n, m = int(page["N"]), int(page["M"])
        if args.method == "uniform":
            preds.append(uniform_splits(n, m))
        else:
            preds.append(random_splits(n, m, rng))

    metrics = eval_predictions(preds, pages)
    result = {"method": args.method, "split": args.split, "seed": args.seed, "metrics": metrics}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_results(f"{args.method}_{args.split}", result, args.output_dir)


if __name__ == "__main__":
    main()

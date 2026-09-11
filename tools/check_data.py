from __future__ import annotations

import argparse
import json

from data_utils import dataset_summary, load_pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/pages.json")
    args = parser.parse_args()
    pages = load_pages(args.data_file)
    print(json.dumps(dataset_summary(pages), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

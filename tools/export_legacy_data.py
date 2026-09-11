"""Export the exact page objects used by the original experiment into data/pages.json.

Run this script inside/against the original experiment directory that contains
`star_baseline_e2e.py`. It calls the original `build_pages()` and
`split_train_val_test()` functions, then applies the same val/test swap used by the
accepted AMVA experiments, and writes a self-contained page-level JSON file.

This keeps the public repository independent from the original absolute paths and
legacy preprocessing code while preserving the exact processed data objects.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _json_scalar(x):
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return x


def clean_page(page: dict, split: str) -> dict:
    n = int(_json_scalar(page["N"]))
    m = int(_json_scalar(page["M"]))
    out = {
        "book_id": str(page.get("book_id", page.get("prj_id", ""))),
        "page_id": str(page.get("page_id", page.get("subprj_id", ""))),
        "N": n,
        "M": m,
        "image_files": [Path(str(x)).name for x in page["image_files"]],
        "candidate_fragments": [str(x) for x in page["candidate_fragments"]],
        "gold_splits": [int(_json_scalar(x)) for x in page["gold_splits"]],
        "split": split,
    }
    # Preserve common identifiers if present.
    for key in ["prj_id", "subprj_id", "prj_name", "subprj_name"]:
        if key in page:
            out[key] = _json_scalar(page[key])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy_root",
        required=True,
        help="Original experiment directory containing star_baseline_e2e.py",
    )
    parser.add_argument("--output", default="data/pages.json")
    parser.add_argument(
        "--legacy_image_dir",
        default=None,
        help="Original block-image directory. Defaults to <legacy_root>/upload_merged",
    )
    parser.add_argument("--output_image_dir", default="data/images")
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy only images referenced by the 397 exported pages.",
    )
    args = parser.parse_args()

    legacy_root = Path(args.legacy_root).resolve()
    sys.path.insert(0, str(legacy_root))
    try:
        from star_baseline_e2e import build_pages, split_train_val_test
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import star_baseline_e2e.py from {legacy_root}"
        ) from exc

    pages = build_pages()
    train_pages, val_pages, test_pages = split_train_val_test(pages)

    # The accepted experiments used the post-swap split:
    # train=296, val=44, test=57.
    val_pages, test_pages = test_pages, val_pages

    exported = []
    for split_name, subset in [
        ("train", train_pages),
        ("val", val_pages),
        ("test", test_pages),
    ]:
        exported.extend(clean_page(page, split_name) for page in subset)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Exported {len(exported)} pages: "
        f"train={len(train_pages)}, val={len(val_pages)}, test={len(test_pages)}"
    )
    print(f"Saved: {out}")

    if args.copy_images:
        source_dir = (
            Path(args.legacy_image_dir).resolve()
            if args.legacy_image_dir
            else legacy_root / "upload_merged"
        )
        target_dir = Path(args.output_image_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        names = sorted({fn for p in exported for fn in p["image_files"]})
        missing = []
        for name in names:
            src = source_dir / name
            dst = target_dir / name
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing.append(name)
        print(f"Copied {len(names) - len(missing)}/{len(names)} referenced images to {target_dir}")
        if missing:
            print(f"WARNING: {len(missing)} images were not found. First 10: {missing[:10]}")


if __name__ == "__main__":
    main()

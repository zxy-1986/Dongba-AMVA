# Data layout

The public code expects the processed page-level annotations used by the paper:

```text
data/
├── pages.json
└── images/
    ├── <block-image-1>.png
    ├── <block-image-2>.png
    ├── <block-image-3>.png
    ├── <block-image-4>.png
    ├── <block-image-5>.png
    └── ...
```

Each entry of `pages.json` has the following minimum fields:

```json
{
  "book_id": "...",
  "page_id": "...",
  "N": 6,
  "M": 26,
  "image_files": ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png", "g.png"],
  "candidate_fragments": ["片段1", "片段2", "..."],
  "gold_splits": [3, 7, 12, 18, 22],
  "split": "test"
}
```

`gold_splits` contains the cumulative text-fragment boundary positions. The split
field is fixed to the book-disjoint split used in the camera-ready paper:
296 train pages / 44 validation pages / 57 test pages.

## Export from the original experiment directory

The repository includes `tools/export_legacy_data.py`. Run it against the original
AMVA experiment directory (the directory containing `star_baseline_e2e.py`):

```bash
python tools/export_legacy_data.py \
  --legacy_root /path/to/original/YOLO \
  --output data/pages.json \
  --copy_images
```

The script calls the original `build_pages()` and `split_train_val_test()` and then
applies the same validation/test swap used by the accepted experiments. This is the
recommended way to preserve the exact preprocessing without keeping absolute paths
or legacy project dependencies in the public repository.

Only redistribute source manuscript images if you have permission to do so.

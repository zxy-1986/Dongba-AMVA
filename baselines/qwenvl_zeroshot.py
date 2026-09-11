"""
Qwen2.5-VL-7B-Instruct zero-shot baseline.

For each image-fragment pair, the model receives a fixed binary yes/no prompt.
The normalized next-token probability P("是") is used as the image-fragment
matching score. Candidate segment scores are obtained by averaging the fragment-level
scores, followed by the same monotonic dynamic-programming decoder used by AMVA.

No autoregressive generation, sampling temperature, top-p, or beam search is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from amva import MAX_SEGMENT_FRAGMENTS, dp_segment
from baselines.common import eval_predictions, get_split, save_results

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWENVL_PATH = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
DEFAULT_DATA_FILE = ROOT / "data" / "pages.json"
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"

PROMPT_TEMPLATE = (
    '这张东巴文图描绘的内容是否对应以下汉译片段?\n\n"{frag}"\n\n'
    '请只回答"是"或"否"。'
)


def load_qwenvl(model_path: str | Path, device: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(model_path))
    return model, processor


@torch.no_grad()
def query_yes_no(
    image: Image.Image,
    fragment: str,
    model,
    processor,
    device: torch.device,
    yes_token_id: int,
    no_token_id: int,
):
    prompt = PROMPT_TEMPLATE.format(frag=fragment[:200])
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    out = model(**inputs)
    next_logits = out.logits[0, -1, :]
    pair = torch.stack([next_logits[yes_token_id], next_logits[no_token_id]])
    probs = F.softmax(pair, dim=-1)
    return probs[0].item()


def predict_one_page(
    page,
    model,
    processor,
    device,
    image_dir: Path,
    yes_token_id,
    no_token_id,
):
    n, m = int(page["N"]), int(page["M"])
    fragments = page["candidate_fragments"]
    if n <= 1 or m <= 1:
        return []

    images = []
    for fn in page["image_files"]:
        path = image_dir / fn
        images.append(Image.open(path).convert("RGB"))

    sim = torch.zeros(n, m, device=device)
    for k in range(n):
        for idx in range(m):
            sim[k, idx] = query_yes_no(
                images[k],
                fragments[idx],
                model,
                processor,
                device,
                yes_token_id,
                no_token_id,
            )

    scores = torch.full((n, m + 1, m + 1), -1e9, device=device)
    for k in range(n):
        cumsum = torch.cat([torch.zeros(1, device=device), sim[k].cumsum(0)])
        for i in range(m):
            for j in range(i + 1, min(m, i + MAX_SEGMENT_FRAGMENTS) + 1):
                scores[k, i, j] = (cumsum[j] - cumsum[i]) / (j - i)

    boundary_logits = torch.zeros(max(m - 1, 0), device=device)
    return dp_segment(scores, boundary_logits, n, m, alpha=0.0)


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL zero-shot baseline.")
    parser.add_argument("--model_path", default=str(DEFAULT_QWENVL_PATH))
    parser.add_argument("--data_file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results/baselines")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, processor = load_qwenvl(args.model_path, args.device)

    tok = processor.tokenizer
    yes_ids = tok.encode("是", add_special_tokens=False)
    no_ids = tok.encode("否", add_special_tokens=False)
    if not yes_ids or not no_ids:
        raise RuntimeError("Tokenizer cannot encode '是'/'否'.")
    yes_token_id, no_token_id = yes_ids[0], no_ids[0]

    pages = get_split(args.data_file, args.split)
    image_dir = Path(args.image_dir)
    preds = []
    for page in tqdm(pages, desc=f"Qwen2.5-VL {args.split}"):
        preds.append(
            predict_one_page(
                page,
                model,
                processor,
                device,
                image_dir,
                yes_token_id,
                no_token_id,
            )
        )

    metrics = eval_predictions(preds, pages)
    result = {
        "baseline": "Qwen2.5-VL-7B zero-shot binary prompting + DP",
        "split": args.split,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    save_results(f"qwenvl_zeroshot_{args.split}", result, args.output_dir)


if __name__ == "__main__":
    main()

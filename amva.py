"""
AMVA: Asymmetric Multi-View Alignment for Dongba Manuscript Image-Text Segment Alignment
=======================================================================================

Paper components:
  1. LoRA domain adaptation on the CN-CLIP vision/text towers.
  2. AMA: five visual spatial views (whole/top/bottom/left/right) with max-view matching.
  3. CABP: context-aware boundary prediction from hand-crafted boundary cues and
     neighboring text embeddings.
  4. Monotonic dynamic-programming decoding with structured alignment learning.

The code also contains the 2^3 ablation settings (E0-E7) and the camera-ready
controlled baselines B0/B1/B2.

All paths are relative by default and can be overridden from the command line.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from data_utils import load_pages, split_pages


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = ROOT / "data" / "pages.json"
DEFAULT_IMAGE_DIR = ROOT / "data" / "images"
DEFAULT_CNCLIP_PATH = ROOT / "models" / "chinese-clip-vit-base-patch16"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

# Training hyperparameters used in the paper.
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0

DIM_EMBED = 512
BOUNDARY_FEAT_DIM = 10
LAMBDA_BOUNDARY = 5.0
ALPHA_BOUNDARY = 1.0
MAX_SEGMENT_FRAGMENTS = 10
TEXT_MAX_LENGTH = 52
CANDIDATE_TEXT_CHAR_LIMIT = 100

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

PUNCT_TYPES = {
    "。": 0,
    "！": 1,
    "？": 2,
    "；": 3,
    "，": 4,
    ",": 4,
    ";": 3,
    "!": 1,
    "?": 2,
}


@dataclass
class ExperimentConfig:
    exp_name: str = "E0_zeroshot"
    use_lora: bool = False
    use_ama: bool = False
    use_cabp: bool = False
    # Camera-ready controlled baselines.
    use_boundary: bool = True
    text_only: bool = False
    seed: int = 42


ALL_EXPERIMENTS: Dict[str, ExperimentConfig] = {
    # Original 2^3 factorial ablation.
    "E0_zeroshot": ExperimentConfig(exp_name="E0_zeroshot"),
    "E1_lora": ExperimentConfig(exp_name="E1_lora", use_lora=True),
    "E2_ama": ExperimentConfig(exp_name="E2_ama", use_ama=True),
    "E3_cabp": ExperimentConfig(exp_name="E3_cabp", use_cabp=True),
    "E4_lora_ama": ExperimentConfig(
        exp_name="E4_lora_ama", use_lora=True, use_ama=True
    ),
    "E5_lora_cabp": ExperimentConfig(
        exp_name="E5_lora_cabp", use_lora=True, use_cabp=True
    ),
    "E6_ama_cabp": ExperimentConfig(
        exp_name="E6_ama_cabp", use_ama=True, use_cabp=True
    ),
    "E7_full": ExperimentConfig(
        exp_name="E7_full", use_lora=True, use_ama=True, use_cabp=True
    ),
    # Camera-ready controlled configurations.
    # B0: strict zero-shot CN-CLIP + monotonic DP, no boundary score.
    "B0_cnclip_dp": ExperimentConfig(
        exp_name="B0_cnclip_dp",
        use_boundary=False,
    ),
    # B1: supervised CN-CLIP + LoRA + structured DP, no AMA/CABP/boundary score.
    "B1_lora_dp": ExperimentConfig(
        exp_name="B1_lora_dp",
        use_lora=True,
        use_boundary=False,
    ),
    # B2: text-only CABP + monotonic DP, no visual matching score.
    "B2_text_cabp_dp": ExperimentConfig(
        exp_name="B2_text_cabp_dp",
        use_cabp=True,
        use_boundary=True,
        text_only=True,
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_punct(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return (
        text.replace(",", "，")
        .replace(";", "；")
        .replace("!", "！")
        .replace("?", "？")
    )


def split_text_with_punct(text: str):
    text = normalize_punct(text)
    parts, puncts = [], []
    current = ""
    for ch in text:
        if ch in "，；。！？":
            if current.strip():
                parts.append(current.strip())
                puncts.append(ch)
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
        puncts.append("")
    return parts, puncts


def extract_boundary_features(fragments: List[str], page_punct_after: List[str]):
    """10-D boundary cues: punctuation one-hot, adjacent lengths, relative position."""
    m = len(fragments)
    feats = np.zeros((max(m - 1, 1), BOUNDARY_FEAT_DIM), dtype=np.float32)
    if m < 2:
        return feats

    lens = [len(f) for f in fragments]
    median_len = np.median(lens) if lens else 1

    for j in range(m - 1):
        punct = page_punct_after[j] if j < len(page_punct_after) else ""
        punct_type = PUNCT_TYPES.get(punct, 5)
        feats[j, punct_type if punct_type < 5 else 5] = 1.0
        feats[j, 6] = lens[j] / max(median_len, 1)
        feats[j, 7] = lens[j + 1] / max(median_len, 1)
        feats[j, 8] = (lens[j] - lens[j + 1]) / max(median_len, 1)
        feats[j, 9] = j / max(m - 1, 1)
    return feats


class AMAPooling(nn.Module):
    """Five fixed spatial views on the CN-CLIP patch-token grid."""

    def __init__(self, num_views: int = 5):
        super().__init__()
        self.num_views = num_views

    def forward(self, hidden_states: torch.Tensor, side: int) -> torch.Tensor:
        patches = hidden_states[:, 1:, :]
        n, p, d = patches.shape
        if side * side != p:
            mean_v = patches.mean(dim=1)
            return mean_v.unsqueeze(1).expand(n, self.num_views, d).contiguous()

        grid = patches.view(n, side, side, d)
        h = side // 2
        view_whole = grid.mean(dim=(1, 2))
        view_top = grid[:, :h, :, :].mean(dim=(1, 2))
        view_bottom = grid[:, h:, :, :].mean(dim=(1, 2))
        view_left = grid[:, :, :h, :].mean(dim=(1, 2))
        view_right = grid[:, :, h:, :].mean(dim=(1, 2))
        return torch.stack(
            [view_whole, view_top, view_bottom, view_left, view_right], dim=1
        )


class AMVA(nn.Module):
    def __init__(self, config: ExperimentConfig, cnclip_path: str | Path):
        super().__init__()
        self.config = config
        self.clip = ChineseCLIPModel.from_pretrained(str(cnclip_path))

        if config.use_lora:
            for p in self.clip.parameters():
                p.requires_grad = False

            lora_vision = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none",
            )
            self.clip.vision_model = get_peft_model(
                self.clip.vision_model, lora_vision
            )

            lora_text = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=["query", "key", "value"],
                bias="none",
            )
            self.clip.text_model = get_peft_model(self.clip.text_model, lora_text)
        else:
            for p in self.clip.parameters():
                p.requires_grad = False

        if config.use_ama:
            self.ama = AMAPooling(num_views=5)

        if config.use_cabp:
            self.cabp_text_proj = nn.Sequential(
                nn.Linear(DIM_EMBED, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            boundary_in_dim = BOUNDARY_FEAT_DIM + 64 * 2 + 1
        else:
            boundary_in_dim = BOUNDARY_FEAT_DIM

        self.boundary_mlp = nn.Sequential(
            nn.Linear(boundary_in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.visual_projection = self.clip.visual_projection

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return [N,512] without AMA or [N,5,512] with AMA."""
        v_out = self.clip.vision_model(pixel_values=pixel_values)
        hidden = v_out.last_hidden_state

        if self.config.use_ama:
            length = hidden.shape[1] - 1
            side = int(length**0.5)
            views = self.ama(hidden, side)
            n5, d_vis = views.shape[0] * views.shape[1], views.shape[2]
            views_flat = views.reshape(n5, d_vis)
            proj = self.visual_projection(views_flat)
            proj = proj.view(views.shape[0], views.shape[1], -1)
            return F.normalize(proj, dim=-1)

        cls = hidden[:, 0, :]
        img_emb = self.visual_projection(cls)
        return F.normalize(img_emb, dim=-1)

    def encode_texts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        if batch == 0:
            return torch.zeros(0, DIM_EMBED, device=input_ids.device)
        out_chunks = []
        for start in range(0, batch, chunk_size):
            end = min(start + chunk_size, batch)
            t_out = self.clip.text_model(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
            )
            pooled = t_out.last_hidden_state[:, 0, :]
            t_emb = self.clip.text_projection(pooled)
            out_chunks.append(t_emb)
        return F.normalize(torch.cat(out_chunks, dim=0), dim=-1)

    def compute_S(
        self,
        visual: torch.Tensor,
        text_grid: torch.Tensor,
        seg_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Candidate-segment cosine score, max over views when AMA is enabled."""
        if self.config.use_ama:
            scores_all = torch.einsum("kvd,ijd->kvij", visual, text_grid)
            scores = scores_all.max(dim=1).values
        else:
            scores = torch.einsum("kd,ijd->kij", visual, text_grid)
        return scores.masked_fill(~seg_mask.unsqueeze(0), -1e9)

    def predict_boundary(
        self,
        boundary_features: torch.Tensor,
        frag_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return boundary logits (sigmoid is applied in the decoder/loss)."""
        if self.config.use_cabp and frag_embeddings is not None:
            m = frag_embeddings.shape[0]
            if m < 2:
                return torch.zeros(0, device=boundary_features.device)
            proj = self.cabp_text_proj(frag_embeddings)
            left, right = proj[:-1], proj[1:]
            cos_sim = (left * right).sum(dim=-1, keepdim=True) / (
                left.norm(dim=-1, keepdim=True)
                * right.norm(dim=-1, keepdim=True)
                + 1e-8
            )
            cabp_feats = torch.cat(
                [boundary_features, left, right, cos_sim], dim=-1
            )
            return self.boundary_mlp(cabp_feats).squeeze(-1)

        return self.boundary_mlp(boundary_features).squeeze(-1)


class AMVADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pages: List[dict],
        processor: ChineseCLIPProcessor,
        image_dir: str | Path,
        image_transform: Optional[Callable[[Image.Image, str], Image.Image]] = None,
    ):
        self.pages = pages
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.pages)

    def _load_image(self, filename: str) -> Image.Image:
        path = self.image_dir / filename
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Cannot load image: {path}") from exc
        if self.image_transform is not None:
            image = self.image_transform(image, filename)
        return image

    def __getitem__(self, idx: int):
        page = self.pages[idx]
        n, m = int(page["N"]), int(page["M"])
        fragments = page["candidate_fragments"]

        images = [self._load_image(fn) for fn in page["image_files"]]
        vis = self.processor(images=images, return_tensors="pt")
        pixel_values = vis["pixel_values"]

        seg_texts = []
        seg_text_indices = torch.full((m + 1, m + 1), -1, dtype=torch.long)
        seg_mask = torch.zeros(m + 1, m + 1, dtype=torch.bool)
        idx_counter = 0
        for i in range(m):
            for j in range(i + 1, min(m, i + MAX_SEGMENT_FRAGMENTS) + 1):
                text = "".join(fragments[i:j])
                if len(text) > CANDIDATE_TEXT_CHAR_LIMIT:
                    text = text[:CANDIDATE_TEXT_CHAR_LIMIT]
                seg_texts.append(text)
                seg_text_indices[i, j] = idx_counter
                seg_mask[i, j] = True
                idx_counter += 1

        if seg_texts:
            txt = self.processor.tokenizer(
                seg_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=TEXT_MAX_LENGTH,
            )
        else:
            txt = {
                "input_ids": torch.zeros(0, 1, dtype=torch.long),
                "attention_mask": torch.zeros(0, 1, dtype=torch.long),
            }

        # Preserve the boundary-feature construction used in the accepted experiments.
        full_text = "".join(fragments)
        _, puncts = split_text_with_punct(full_text)
        if len(puncts) < len(fragments):
            puncts = puncts + [""] * (len(fragments) - len(puncts))
        elif len(puncts) > len(fragments):
            puncts = puncts[: len(fragments)]
        boundary_feat = extract_boundary_features(fragments, puncts)
        boundary_feat_tensor = torch.from_numpy(boundary_feat).float()

        gold_splits = [int(x) for x in page["gold_splits"]]
        boundary_labels = torch.zeros(max(m - 1, 1), dtype=torch.float)
        for split_pos in gold_splits:
            idx_b = split_pos - 1
            if 0 <= idx_b < m - 1:
                boundary_labels[idx_b] = 1.0

        return {
            "pixel_values": pixel_values,
            "text_input_ids": txt["input_ids"],
            "text_attention_mask": txt["attention_mask"],
            "seg_mask": seg_mask,
            "seg_text_indices": seg_text_indices,
            "boundary_features": boundary_feat_tensor,
            "boundary_labels": boundary_labels,
            "N": n,
            "M": m,
            "gold_splits": torch.tensor(gold_splits, dtype=torch.long),
        }


def dp_segment(
    scores: torch.Tensor,
    boundary_logits: torch.Tensor,
    n: int,
    m: int,
    alpha: float = ALPHA_BOUNDARY,
):
    """Monotonic Viterbi-style decoder."""
    device = scores.device
    neg_inf = -1e18
    dp = torch.full((n + 1, m + 1), neg_inf, device=device)
    backptr = torch.full(
        (n + 1, m + 1), -1, dtype=torch.long, device=device
    )
    dp[0, 0] = 0.0
    b_sig = torch.sigmoid(boundary_logits)

    for k in range(1, n + 1):
        j_min = k
        j_max = m - (n - k)
        for j in range(j_min, j_max + 1):
            best_score, best_i = neg_inf, -1
            for i in range(k - 1, j):
                if j - i > MAX_SEGMENT_FRAGMENTS:
                    continue
                prev = dp[k - 1, i].item()
                if prev <= neg_inf / 2:
                    continue
                seg_score = scores[k - 1, i, j].item()
                if seg_score <= -1e8:
                    continue
                boundary_bonus = 0.0
                if k > 1 and 0 <= i - 1 < len(b_sig):
                    boundary_bonus = alpha * b_sig[i - 1].item()
                total = prev + seg_score + boundary_bonus
                if total > best_score:
                    best_score, best_i = total, i
            dp[k, j] = best_score
            backptr[k, j] = best_i

    splits = []
    k, j = n, m
    while k > 1:
        i = backptr[k, j].item()
        if i < 0:
            i = max(0, j - 1)
        splits.append(i)
        k -= 1
        j = i
    splits.reverse()
    return splits


def gold_path_score(
    scores: torch.Tensor,
    boundary_logits: torch.Tensor,
    gold_splits,
    n: int,
    m: int,
    alpha: float = ALPHA_BOUNDARY,
):
    boundaries = [0] + [int(x) for x in gold_splits] + [m]
    total = 0.0
    for k in range(n):
        i, j = boundaries[k], boundaries[k + 1]
        if j > i and (j - i) <= MAX_SEGMENT_FRAGMENTS:
            total = total + scores[k, i, j]
            if k > 0 and 0 <= i - 1 < len(boundary_logits):
                total = total + alpha * torch.sigmoid(boundary_logits[i - 1])
        else:
            return None
    return total


def dp_total_logsumexp(
    scores: torch.Tensor,
    boundary_logits: torch.Tensor,
    n: int,
    m: int,
    alpha: float = ALPHA_BOUNDARY,
):
    device = scores.device
    neg_inf = -1e18
    log_dp = torch.full((n + 1, m + 1), neg_inf, device=device)
    log_dp[0, 0] = 0.0
    b_sig = torch.sigmoid(boundary_logits)

    for k in range(1, n + 1):
        j_min = k
        j_max = m - (n - k)
        for j in range(j_min, j_max + 1):
            terms = []
            for i in range(k - 1, j):
                if j - i > MAX_SEGMENT_FRAGMENTS:
                    continue
                prev = log_dp[k - 1, i]
                if prev <= neg_inf / 2:
                    continue
                seg_score = scores[k - 1, i, j]
                if seg_score <= -1e8:
                    continue
                boundary_bonus = 0.0
                if k > 1 and 0 <= i - 1 < len(b_sig):
                    boundary_bonus = alpha * b_sig[i - 1]
                terms.append(prev + seg_score + boundary_bonus)
            if terms:
                log_dp[k, j] = torch.logsumexp(torch.stack(terms), dim=0)
    return log_dp[n, m]


def alignment_loss(
    scores: torch.Tensor,
    boundary_logits: torch.Tensor,
    gold_splits,
    n: int,
    m: int,
    alpha: float = ALPHA_BOUNDARY,
):
    gold = gold_path_score(
        scores, boundary_logits, gold_splits, n, m, alpha=alpha
    )
    if gold is None:
        return None
    total = dp_total_logsumexp(
        scores, boundary_logits, n, m, alpha=alpha
    )
    return -(gold - total)


def boundary_loss(boundary_logits, boundary_labels):
    if len(boundary_logits) == 0:
        return torch.tensor(0.0, device=boundary_labels.device)
    return F.binary_cross_entropy_with_logits(boundary_logits, boundary_labels)


def forward_page(model: AMVA, item: dict, device: torch.device):
    pixel_values = item["pixel_values"].to(device)
    text_input_ids = item["text_input_ids"].to(device)
    text_attention_mask = item["text_attention_mask"].to(device)
    seg_mask = item["seg_mask"].to(device)
    seg_text_indices = item["seg_text_indices"].to(device)
    boundary_feats = item["boundary_features"].to(device)
    n, m = item["N"], item["M"]

    visual = model.encode_images(pixel_values)
    text_seq = model.encode_texts(text_input_ids, text_attention_mask)

    text_grid = torch.zeros(m + 1, m + 1, DIM_EMBED, device=device)
    valid_mask = seg_text_indices >= 0
    valid_idx = seg_text_indices[valid_mask]
    text_grid[valid_mask] = text_seq[valid_idx]

    if model.config.text_only:
        scores = torch.zeros(
            n, m + 1, m + 1, device=device, dtype=text_grid.dtype
        )
        scores = scores.masked_fill(~seg_mask.unsqueeze(0), -1e9)
    else:
        scores = model.compute_S(visual, text_grid, seg_mask)

    if model.config.use_cabp:
        frag_embeddings = torch.stack(
            [text_grid[idx, idx + 1] for idx in range(m)], dim=0
        )
        boundary_logits = model.predict_boundary(
            boundary_feats, frag_embeddings=frag_embeddings
        )
    else:
        boundary_logits = model.predict_boundary(boundary_feats)

    return scores, boundary_logits


def evaluate(model: AMVA, dataset: AMVADataset, device: torch.device):
    model.eval()
    correct_splits = total_splits = 0
    page_perfect = n_pages = 0
    by_nm = defaultdict(
        lambda: {"correct": 0, "total": 0, "pages": 0, "perfect": 0}
    )
    b_correct = b_total = 0

    with torch.no_grad():
        for item in tqdm(dataset, desc="Eval", leave=False):
            n, m = item["N"], item["M"]
            gold = item["gold_splits"].tolist()
            scores, boundary_logits = forward_page(model, item, device)
            alpha = ALPHA_BOUNDARY if model.config.use_boundary else 0.0
            pred = dp_segment(scores, boundary_logits, n, m, alpha=alpha)

            correct = sum(int(p == g) for p, g in zip(pred, gold))
            correct_splits += correct
            total_splits += len(gold)
            if correct == len(gold):
                page_perfect += 1
            n_pages += 1

            key = "N=M" if n == m else ("N<M" if n < m else "N>M")
            by_nm[key]["correct"] += correct
            by_nm[key]["total"] += len(gold)
            by_nm[key]["pages"] += 1
            if correct == len(gold):
                by_nm[key]["perfect"] += 1

            if model.config.use_boundary and len(boundary_logits) >= n - 1 and n > 1:
                topk = torch.topk(boundary_logits, n - 1).indices.cpu().tolist()
                topk_set = {idx + 1 for idx in topk}
                gold_set = set(gold)
                b_correct += len(topk_set & gold_set)
                b_total += len(gold_set)

    return {
        "split_acc": correct_splits / max(total_splits, 1),
        "page_perfect_rate": page_perfect / max(n_pages, 1),
        "n_pages": n_pages,
        "by_NM": dict(by_nm),
        "boundary_only_acc": b_correct / max(b_total, 1),
    }


def build_datasets(
    data_file: str | Path,
    image_dir: str | Path,
    processor: ChineseCLIPProcessor,
    image_transform=None,
):
    pages = load_pages(data_file)
    train_pages, val_pages, test_pages = split_pages(pages)
    return (
        AMVADataset(train_pages, processor, image_dir, image_transform),
        AMVADataset(val_pages, processor, image_dir, image_transform),
        AMVADataset(test_pages, processor, image_dir, image_transform),
    )


def train_one_experiment(
    config: ExperimentConfig,
    output_dir: str | Path,
    data_file: str | Path = DEFAULT_DATA_FILE,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    cnclip_path: str | Path = DEFAULT_CNCLIP_PATH,
    device_str: str = "cuda:0",
):
    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pages(data_file)
    train_pages, val_pages, test_pages = split_pages(pages)
    print(
        f"Dataset: train={len(train_pages)}, val={len(val_pages)}, test={len(test_pages)}"
    )

    device = torch.device(device_str)
    processor = ChineseCLIPProcessor.from_pretrained(str(cnclip_path))
    model = AMVA(config, cnclip_path).to(device)

    train_ds = AMVADataset(train_pages, processor, image_dir)
    val_ds = AMVADataset(val_pages, processor, image_dir)
    test_ds = AMVADataset(test_pages, processor, image_dir)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable) / 1e6
    print(f"Experiment: {config.exp_name}; trainable parameters: {n_params:.4f}M")

    is_zeroshot = (
        not config.use_lora and not config.use_ama and not config.use_cabp
    )
    if is_zeroshot:
        train_metrics = evaluate(model, train_ds, device)
        val_metrics = evaluate(model, val_ds, device)
        test_metrics = evaluate(model, test_ds, device)
        _save_results(
            config,
            train_metrics,
            val_metrics,
            test_metrics,
            None,
            n_params,
            output_dir,
            mode="zero_shot",
        )
        _print_results(config, train_metrics, val_metrics, test_metrics)
        return

    if not trainable:
        raise RuntimeError("No trainable parameters for this configuration.")
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_align_loss = train_bd_loss = 0.0
        n_batches = n_skipped = n_oom = 0

        indices = np.random.permutation(len(train_ds))
        for idx in tqdm(indices, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
            try:
                item = train_ds[int(idx)]
                boundary_labels = item["boundary_labels"].to(device)
                n, m = item["N"], item["M"]
                gold = item["gold_splits"].to(device)

                scores, boundary_logits = forward_page(model, item, device)
                alpha = ALPHA_BOUNDARY if config.use_boundary else 0.0
                l_align = alignment_loss(
                    scores, boundary_logits, gold, n, m, alpha=alpha
                )
                if l_align is None:
                    n_skipped += 1
                    continue

                if config.use_boundary:
                    l_bd = boundary_loss(boundary_logits, boundary_labels)
                    loss = l_align + LAMBDA_BOUNDARY * l_bd
                else:
                    l_bd = torch.tensor(0.0, device=device)
                    loss = l_align

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                optimizer.step()

                train_align_loss += float(l_align.item())
                train_bd_loss += float(l_bd.item())
                n_batches += 1
            except torch.cuda.OutOfMemoryError:
                n_oom += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        train_align_loss /= max(n_batches, 1)
        train_bd_loss /= max(n_batches, 1)
        val_metrics = evaluate(model, val_ds, device)
        print(
            f"Epoch {epoch + 1}: L_align={train_align_loss:.3f} "
            f"L_bd={train_bd_loss:.3f} skip={n_skipped} oom={n_oom} | "
            f"Val SplitAcc={val_metrics['split_acc']:.4f}"
        )

        if val_metrics["split_acc"] > best_val:
            best_val = val_metrics["split_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint.")

    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_ds, device)
    val_metrics = evaluate(model, val_ds, device)
    test_metrics = evaluate(model, test_ds, device)

    _save_results(
        config,
        train_metrics,
        val_metrics,
        test_metrics,
        best_state,
        n_params,
        output_dir,
        mode="trained",
        best_val=best_val,
    )
    _print_results(config, train_metrics, val_metrics, test_metrics)


def load_trained_model(
    exp_name: str,
    checkpoint: str | Path,
    cnclip_path: str | Path,
    device: torch.device,
    seed: int = 42,
):
    if exp_name not in ALL_EXPERIMENTS:
        raise KeyError(f"Unknown experiment: {exp_name}")
    config = replace(ALL_EXPERIMENTS[exp_name], seed=seed)
    model = AMVA(config, cnclip_path).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _print_results(config, train_m, val_m, test_m):
    n_lt_m = test_m.get("by_NM", {}).get("N<M", {})
    print(f"\n{config.exp_name}")
    print(f"  Train SplitAcc: {train_m['split_acc']:.4f}")
    print(f"  Val   SplitAcc: {val_m['split_acc']:.4f}")
    print(f"  Test  SplitAcc: {test_m['split_acc']:.4f}")
    print(f"  Test  Page-EM:  {test_m['page_perfect_rate']:.4f}")
    if n_lt_m:
        acc = n_lt_m.get("correct", 0) / max(n_lt_m.get("total", 1), 1)
        print(f"  Test  N<M Acc:  {acc:.4f}")


def _save_results(
    config,
    train_m,
    val_m,
    test_m,
    best_state,
    n_params,
    output_dir,
    mode="trained",
    best_val=None,
):
    results = {
        "exp_name": config.exp_name,
        "config": {
            "use_lora": config.use_lora,
            "use_ama": config.use_ama,
            "use_cabp": config.use_cabp,
            "use_boundary": config.use_boundary,
            "text_only": config.text_only,
            "seed": config.seed,
        },
        "train": train_m,
        "val": val_m,
        "test": test_m,
        "n_params_M": n_params,
        "mode": mode,
    }
    if best_val is not None:
        results["best_val_during_training"] = best_val

    output_dir = Path(output_dir)
    with (output_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=float)
    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate AMVA configurations.")
    parser.add_argument("--exp", required=True, choices=sorted(ALL_EXPERIMENTS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--cnclip_path", default=str(DEFAULT_CNCLIP_PATH))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = replace(ALL_EXPERIMENTS[args.exp], seed=args.seed)
    output_dir = Path(args.output_root) / f"{args.exp}_seed{args.seed}"
    train_one_experiment(
        config=config,
        output_dir=output_dir,
        data_file=args.data_file,
        image_dir=args.image_dir,
        cnclip_path=args.cnclip_path,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

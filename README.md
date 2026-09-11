Dongba-AMVA

Official reproduction code for AMVA (Asymmetric Multi-View Alignment) for
image-text segment alignment in Dongba manuscripts.

Paper title (Chinese): 面向东巴古籍图文段对齐的非对称多视图融合与上下文边界预测方法.

AMVA addresses the common one-to-many structure in Dongba manuscripts, where one
pictographic image block corresponds to multiple consecutive literal-translation
fragments. The implementation contains:

LoRA domain adaptation on the CN-CLIP vision and text towers;

AMA, a five-view asymmetric visual aggregation module (whole/top/bottom/left/right);

CABP, context-aware boundary prediction using adjacent text semantics and boundary cues;

monotonic dynamic programming for globally structured decoding.

The repository also includes the full E0-E7 factorial ablation, camera-ready
controlled baselines B0/B1/B2, the Qwen2.5-VL zero-shot baseline, and robustness
evaluation.

Repository structure

Dongba-AMVA/
├── README.md
├── LICENSE
├── requirements.txt
├── amva.py                         # AMVA, training, E0-E7 and B0/B1/B2
├── evaluate.py                     # checkpoint evaluation
├── robustness_eval.py              # blur / fading / local-damage tests
├── data_utils.py                    # processed-data loading and validation
├── baselines/
│   ├── __init__.py
│   ├── common.py
│   ├── rule_baselines.py           # Random Split / Uniform Split
│   └── qwenvl_zeroshot.py          # Qwen2.5-VL-7B binary-prompt + DP
├── tools/
│   ├── export_legacy_data.py       # exports exact processed pages from old project
│   └── check_data.py
├── data/
│   ├── README.md
│   ├── pages.json                  # add this after export
│   └── images/                     # add referenced block images
├── models/
│   └── README.md
├── results/
│   └── reported_results.json       # numbers reported in the camera-ready paper
└── outputs/                        # generated checkpoints/results (gitignored)

Installation

Python 3.10+ is recommended.

pip install -r requirements.txt

The camera-ready reproduction used CN-CLIP ViT-B/16. Put the pretrained checkpoint
under:

models/chinese-clip-vit-base-patch16/

For the optional Qwen baseline, put the checkpoint under:

models/Qwen2.5-VL-7B-Instruct/

All model/data paths can also be overridden through command-line arguments, so no
absolute server paths are required.

Data preparation

The public code uses a processed page-level file data/pages.json and the image-block
directory data/images/.

To export exactly the data objects used by the original accepted experiments, run the
provided exporter against the old experiment directory containing
star_baseline_e2e.py:

python tools/export_legacy_data.py \
  --legacy_root /path/to/original/YOLO \
  --output data/pages.json \
  --copy_images

This exporter calls the original build_pages() and split_train_val_test(), applies
the same final validation/test assignment used by the paper, and removes absolute
paths from image filenames.

Expected final split:

Split

Books

Pages

Train

15

296

Validation

3

44

Test

3

57

Check the exported file before training:

python tools/check_data.py --data_file data/pages.json

See data/README.md for the JSON schema.

Train the full AMVA model

python amva.py \
  --exp E7_full \
  --seed 42 \
  --device cuda:0

Outputs are saved by default to:

outputs/E7_full_seed42/
├── best_model.pt
└── results.json

The best checkpoint is selected by validation SplitAcc with early stopping patience 8.

Full factorial ablation

The original 2^3 ablation settings are directly available:

python amva.py --exp E0_zeroshot --seed 42
python amva.py --exp E1_lora --seed 42
python amva.py --exp E2_ama --seed 42
python amva.py --exp E3_cabp --seed 42
python amva.py --exp E4_lora_ama --seed 42
python amva.py --exp E5_lora_cabp --seed 42
python amva.py --exp E6_ama_cabp --seed 42
python amva.py --exp E7_full --seed 42

Camera-ready controlled baselines

# B0: strict zero-shot CN-CLIP + monotonic DP; no boundary predictor
python amva.py --exp B0_cnclip_dp --seed 42

# B1: supervised CN-CLIP + LoRA + DP; no AMA/CABP/boundary score
python amva.py --exp B1_lora_dp --seed 42

# B2: text-only CABP + DP; no visual matching score
python amva.py --exp B2_text_cabp_dp --seed 42

Evaluate a saved checkpoint

python evaluate.py \
  --exp E7_full \
  --checkpoint outputs/E7_full_seed42/best_model.pt \
  --split test \
  --device cuda:0

Robustness evaluation

The camera-ready paper reports three test-time corruptions without retraining:

Gaussian blur: radius = 2.0;

ink fading: blend with white background at 0.40;

local damage: deterministic random erasure of approximately 6% image area.

Run:

python robustness_eval.py \
  --exp E7_full \
  --checkpoint outputs/E7_full_seed42/best_model.pt \
  --device cuda:0

Rule baselines

python -m baselines.rule_baselines --method random --split test --seed 42
python -m baselines.rule_baselines --method uniform --split test

Qwen2.5-VL-7B zero-shot baseline

The Qwen baseline uses a fixed binary prompt for every image-fragment pair. It reads
the next-token logits of Chinese tokens “是” and “否”, normalizes the two logits with
softmax, and uses P(是) as the matching score. Candidate segment scores are the mean
of their internal fragment scores, followed by the same monotonic DP decoder. No
autoregressive generation or sampling is used.

python -m baselines.qwenvl_zeroshot \
  --split test \
  --device cuda:0

Key hyperparameters

Hyperparameter

Value

CN-CLIP input size

224 × 224

LoRA rank

8

LoRA alpha

16

LoRA dropout

0.05

Learning rate

1e-4

Weight decay

1e-4

Max epochs

30

Early stopping patience

8

Gradient clipping

1.0

Boundary loss weight

5.0

Boundary decoding weight

1.0

Maximum candidate segment length

10 fragments

Text tokenizer max length

52

Default random seed

42

Reported results

The exact numbers reported in the camera-ready paper are stored in
results/reported_results.json.

Full AMVA (E7, seed 42):

SplitAcc

Page-EM

N<M Acc

63.83

40.35

51.12

Reproducibility notes

Use the exported pages.json rather than re-running a new random split.

The released split is book-disjoint to avoid manuscript-level style leakage.

The candidate-segment upper bound is fixed at 10, matching the accepted experiments.

For B0/B1, boundary scores are fully disabled in both training and decoding.

CABP outputs boundary logits; sigmoid is applied only when the boundary probability
is consumed by the decoder/loss.

Data and copyright

The code is released under the MIT License. Data/images may have different copyright
or redistribution requirements depending on their source. Only publish manuscript
images if you have the corresponding redistribution permission. The processed
annotation file and split metadata can be released separately from the images if
necessary.

Citation

Please cite the camera-ready paper after the final bibliographic information is available.
A BibTeX entry can then be added here.

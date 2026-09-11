#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
DEVICE="${2:-cuda:0}"

for EXP in \
  E0_zeroshot E1_lora E2_ama E3_cabp \
  E4_lora_ama E5_lora_cabp E6_ama_cabp E7_full
 do
  python amva.py --exp "$EXP" --seed "$SEED" --device "$DEVICE"
 done

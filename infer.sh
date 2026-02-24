#!/usr/bin/env bash
set -euo pipefail

## Inference ##

stage="pre_ytb_dav"
data_path="./datasets/Demo"
model="r50_aotl"
ckpt_path="./pretrain_models/R50_AOTL_PRE_YTB_DAV.pth"

python tools/demo.py --amp \
    --stage "${stage}" \
    --model "${model}" \
    --data_path "${data_path}" \
    --ckpt_path "${ckpt_path}"
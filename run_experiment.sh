#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

dataset="Cora"
output_file="${script_dir}/out/results.csv"
data_root="${script_dir}/data"

python "${script_dir}/main.py" \
    "${dataset}" \
    "${output_file}" \
    --data-root "${data_root}" \
    --conv-layer-type GCNConv \
    --hidden-channels 128 \
    --out-channels 64 \
    --reduction-method pca \
    --query-size 50 \
    --train-ratio 0.8 \
    --epochs 100 \
    --seed 42 \
    --device cpu \
    --normalize-features \
    --verbose

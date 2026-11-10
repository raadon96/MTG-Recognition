#!/usr/bin/env bash
# Throwaway (#198): the confirm half, one run at a time so the times compare, each
# model with its starting settings and its chosen ones; then the thread count.
#   bash scripts/ocr_settings_confirm.sh
unset VIRTUAL_ENV
set -u
ONNX=models/ocr/test/onnx
V6S="--det PP-OCRv6_small_det=$ONNX/PP-OCRv6_small_det --rec PP-OCRv6_small_rec=$ONNX/PP-OCRv6_small_rec --det-thresh 0.2 --det-box-thresh 0.45 --det-unclip 1.4"
FAST="--det PP-OCRv5_mobile_det=$ONNX/PP-OCRv5_mobile_det --rec PP-OCRv6_tiny_rec=$ONNX/PP-OCRv6_tiny_rec --det-thresh 0.05 --det-box-thresh 0.6 --det-unclip 1.5"
SMALLER="--det-limit-type max --det-limit-side-len 960 --min-dim 64"
runs=(
  "c-v6s-start|4|$V6S"
  "c-v6s-chosen|4|$V6S $SMALLER"
  "c-fast-start|4|$FAST"
  "c-fast-smaller|4|$FAST $SMALLER"
)
for t in 1 2 8 16; do
  runs+=("c-v6s-chosen-t$t|$t|$V6S $SMALLER" "c-fast-start-t$t|$t|$FAST")
done
mkdir -p output/ocr_compare/logs
for entry in "${runs[@]}"; do
  IFS='|' read -r name threads flags <<< "$entry"
  [ -f "output/ocr_compare/$name/summary.json" ] && continue
  repeats=3; [[ $name == *-t* ]] && repeats=1
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/ocr_compare.py run "$name" --subset confirm \
    --onnxruntime --no-fp16 --threads "$threads" --repeats $repeats $flags \
    > "output/ocr_compare/logs/$name.log" 2>&1
done
echo done

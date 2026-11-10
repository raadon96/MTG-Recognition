#!/usr/bin/env bash
unset VIRTUAL_ENV
# Throwaway (#198): one ocr_compare.py run per setting value on the tune half,
# one factor at a time from each model's starting point, on onnxruntime with 4
# threads. Accuracy only: three runs share the CPU, so their times aren't compared.
#   bash scripts/ocr_settings_sweep.sh
set -u
ONNX=models/ocr/test/onnx
V6S="--det PP-OCRv6_small_det=$ONNX/PP-OCRv6_small_det --rec PP-OCRv6_small_rec=$ONNX/PP-OCRv6_small_rec"
FAST="--det PP-OCRv5_mobile_det=$ONNX/PP-OCRv5_mobile_det --rec PP-OCRv6_tiny_rec=$ONNX/PP-OCRv6_tiny_rec"
V6S_BASE="--det-thresh 0.2 --det-box-thresh 0.45 --det-unclip 1.4"
FAST_BASE="--det-thresh 0.05 --det-box-thresh 0.6 --det-unclip 1.5"

runs=(
  # round 1: one factor at a time
  "s-v6s-thresh0.05|$V6S $V6S_BASE --det-thresh 0.05"
  "s-v6s-thresh0.1|$V6S $V6S_BASE --det-thresh 0.1"
  "s-v6s-thresh0.3|$V6S $V6S_BASE --det-thresh 0.3"
  "s-v6s-box0.3|$V6S $V6S_BASE --det-box-thresh 0.3"
  "s-v6s-box0.6|$V6S $V6S_BASE --det-box-thresh 0.6"
  "s-v6s-unclip1.2|$V6S $V6S_BASE --det-unclip 1.2"
  "s-v6s-unclip1.6|$V6S $V6S_BASE --det-unclip 1.6"
  "s-v6s-unclip2.0|$V6S $V6S_BASE --det-unclip 2.0"
  "s-v6s-limit960max|$V6S $V6S_BASE --det-limit-side-len 960 --det-limit-type max"
  "s-v6s-limit736min|$V6S $V6S_BASE --det-limit-side-len 736 --det-limit-type min"
  "s-v6s-height32|$V6S $V6S_BASE --target-height 32"
  "s-v6s-height40|$V6S $V6S_BASE --target-height 40"
  "s-v6s-height64|$V6S $V6S_BASE --target-height 64"
  "s-fast-thresh0.1|$FAST $FAST_BASE --det-thresh 0.1"
  "s-fast-thresh0.2|$FAST $FAST_BASE --det-thresh 0.2"
  "s-fast-thresh0.3|$FAST $FAST_BASE --det-thresh 0.3"
  "s-fast-box0.4|$FAST $FAST_BASE --det-box-thresh 0.4"
  "s-fast-box0.5|$FAST $FAST_BASE --det-box-thresh 0.5"
  "s-fast-unclip1.4|$FAST $FAST_BASE --det-unclip 1.4"
  "s-fast-unclip1.8|$FAST $FAST_BASE --det-unclip 1.8"
  "s-fast-unclip2.0|$FAST $FAST_BASE --det-unclip 2.0"
  "s-fast-limit960max|$FAST $FAST_BASE --det-limit-side-len 960 --det-limit-type max"
  "s-fast-limit736min|$FAST $FAST_BASE --det-limit-side-len 736 --det-limit-type min"
  "s-fast-height32|$FAST $FAST_BASE --target-height 32"
  "s-fast-height40|$FAST $FAST_BASE --target-height 40"
  "s-fast-height64|$FAST $FAST_BASE --target-height 64"
  # round 2: a smaller detection input for the collector information, the name
  # crop kept at 64 px as the default min/64 gave it
  "s2-v6s-max640-min64|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 640 --min-dim 64"
  "s2-v6s-max800-min64|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 800 --min-dim 64"
  "s2-v6s-max960-min64|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 960 --min-dim 64"
  "s2-v6s-max1280-min64|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 1280 --min-dim 64"
  "s2-v6s-max960-min64-height40|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 960 --min-dim 64 --target-height 40"
  "s2-v6s-max960-min64-height64|$V6S $V6S_BASE --det-limit-type max --det-limit-side-len 960 --min-dim 64 --target-height 64"
  "s2-fast-max640-min64|$FAST $FAST_BASE --det-limit-type max --det-limit-side-len 640 --min-dim 64"
  "s2-fast-max800-min64|$FAST $FAST_BASE --det-limit-type max --det-limit-side-len 800 --min-dim 64"
  "s2-fast-max960-min64|$FAST $FAST_BASE --det-limit-type max --det-limit-side-len 960 --min-dim 64"
  "s2-fast-max1280-min64|$FAST $FAST_BASE --det-limit-type max --det-limit-side-len 1280 --min-dim 64"
  "s2-fast-max960-min64-height64|$FAST $FAST_BASE --det-limit-type max --det-limit-side-len 960 --min-dim 64 --target-height 64"
)
mkdir -p output/ocr_compare/logs
for entry in "${runs[@]}"; do
  name=${entry%%|*}
  [ -f "output/ocr_compare/$name/summary.json" ] && continue
  # shellcheck disable=SC2086
  uv run --no-sync python scripts/ocr_compare.py run "$name" --subset tune \
    --onnxruntime --no-fp16 --threads 4 ${entry#*|} \
    > "output/ocr_compare/logs/$name.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
done
wait
echo done

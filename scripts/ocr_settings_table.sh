#!/usr/bin/env bash
# Throwaway (#198): candidate-driven results ("candidate, shown") and OCR time per
# run, for the given subset. Usage: bash scripts/ocr_settings_table.sh tune RUN...
unset VIRTUAL_ENV
subset=$1; shift
for run in "$@"; do
  row=$(uv run --no-sync python scripts/extraction.py report "$run" --subset "$subset" \
    --variant "candidate, shown" 2>/dev/null | grep "^| candidate" | head -1)
  t=$(python3 -c "import json;d=json.load(open('output/ocr_compare/$run/summary.json'))['timing'];print(d['total']['median_s'],d['detection']['median_s'],d['recognition']['median_s'])")
  echo "| $run ${row#| candidate, shown} $t |"
done

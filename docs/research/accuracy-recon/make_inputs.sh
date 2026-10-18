#!/bin/bash
# usage: make_inputs.sh — symlink every UUID-named test image into /tmp/acc6/in/<category>/ (input for run.sh)
cd /home/raadon96/Insync/Programming/decksray/MTG-Recognition
R=/tmp/acc6/in
UUID_RE='^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.jpg$'
while read -r cat src; do
  mkdir -p $R/$cat
  for f in $src/*; do
    basename "$f" | grep -Eq "$UUID_RE" && ln -sf "$PWD/$f" "$R/$cat/"
  done
done <<MAP
1_photos_single_card data/test/1_photos_single_card/images
4_scans_single_card_no_bg data/test/4_scans_single_card_no_bg/images
6_benchmark__ocr__common_layout data/test/6_benchmark/ocr/common_layout
6_benchmark__ocr__uncommon_layout data/test/6_benchmark/ocr/uncommon_layout
6_benchmark__raw data/test/6_benchmark/images/raw
MAP

#!/bin/bash
# usage: run.sh <run-name>
cd /home/raadon96/Insync/Programming/decksray/MTG-Recognition
R=/tmp/acc6; O=$R/$1; mkdir -p $O
for d in $R/in/*; do c=$(basename $d)
  s=$(date +%s.%N)
  uv run python -m mtg_recognition.engine.pipeline -i $d -o $O/$c.json > $O/$c.log 2>&1
  e=$(date +%s.%N); echo "$c $(ls $d | wc -l) $(echo "$e - $s" | bc)" >> $O/times.txt
done

#!/usr/bin/env bash
# Checks a built image: the web app answers Streamlit's health check, and
# `mtg recognize` returns the printing of a committed query image.
#
# Usage, from the repository root: check-image.sh <image>
set -euo pipefail

image=$1
# Named by its printing's UUID. Brimstone Trebuchet: the benchmark recognises
# it, and among the fixtures it has the widest score gap to the second printing.
fixture=tests/regression/matching/fixtures/74b27345-3ebb-53c3-9587-c86f2c74b154.jpg
expected=$(basename "$fixture" .jpg)
health_timeout_s=90
port=8501

container=$(docker run -d -p "$port:$port" "$image")
trap 'docker rm -f "$container" >/dev/null' EXIT

echo "Waiting up to ${health_timeout_s} s for the web app's health check"
deadline=$((SECONDS + health_timeout_s))
until curl -fsS --max-time 5 "http://localhost:$port/_stcore/health" >/dev/null 2>&1; do
  running=$(docker inspect -f '{{.State.Running}}' "$container")
  if [[ $running != true ]]; then
    echo "::error::The container stopped before the web app answered its health check"
    docker logs "$container"
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "::error::The web app didn't answer its health check within ${health_timeout_s} s"
    docker logs "$container"
    exit 1
  fi
  sleep 2
done
echo "The web app answered its health check"

# The JSON goes to stdout and everything else to stderr, so the result needs no
# writable mount: the image runs as a user that can't write to the runner's
# directories.
result=$(mktemp)
if ! docker run --rm -v "$PWD/$fixture:/query/$expected.jpg:ro" "$image" \
  sh -c 'mtg recognize -i "$1" -o /tmp/result.json >&2 && cat /tmp/result.json' \
  sh "/query/$expected.jpg" >"$result"; then
  echo "::error::mtg recognize failed"
  exit 1
fi

got=$(jq -r '.[0].cards[0].ranked[0].uuid // "none"' "$result")
if [[ $got != "$expected" ]]; then
  echo "::error::mtg recognize returned printing $got, expected $expected"
  cat "$result"
  exit 1
fi
echo "mtg recognize returned the fixture's printing, $expected"

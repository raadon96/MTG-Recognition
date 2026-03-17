#!/bin/sh
set -eu

ASSET_CACHE_DIR="${ULTRALYTICS_ASSET_CACHE_DIR:-/home/appuser/.cache/ultralytics-assets}"
# Resolve cache dir:
# - If env var ULTRALYTICS_ASSET_CACHE_DIR is set, use that value.
# - Otherwise (:-...), use default /home/appuser/.cache/ultralytics-assets.

mkdir -p "$ASSET_CACHE_DIR"
# Ensure that dir exists.

for file in mobileclip_blt.ts mobileclip2_b.ts; do
    cached="$ASSET_CACHE_DIR/$file"
    target="/app/$file"

    if [ ! -e "$target" ]; then
        ln -s "$cached" "$target"
    fi
    # If target /app/<file> does not exist, create symlink
done

exec "$@"

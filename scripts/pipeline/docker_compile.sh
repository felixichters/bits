#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/felix/projects/reveng/25WS-RevEng/"
DATA_DIR="/home/felix/projects/reveng/final/data/"
BUILD_DIR="/home/felix/projects/reveng/final/build/"

docker run -it --rm \
  --cpus=40 \
  -v "${DATA_DIR}:/data" \
  -v "${BUILD_DIR}:/build" \
  -v "${REPO_ROOT}:/repo" \
  reveng-compile \
  uv run python scripts/pipeline/compile_all.py \
    --package-list /data/manifests/package_list.txt \
    --build-root /build \
    --workers 8 \
    --resume \
    --status-file /data/manifests/compile_status.json

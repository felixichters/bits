#!/bin/bash
set -euo pipefail

REPO_ROOT="/home/felix/projects/reveng/25WS-RevEng/"
DATA_DIR="/home/felix/projects/reveng/final/data/"
BUILD_DIR="/home/felix/projects/reveng/final/build/"
BINARIES_DIR="/mnt/hdd/reveng-binaries/"

docker run -it --rm \
  -v "${DATA_DIR}:/data" \
  -v "${BUILD_DIR}:/build" \
  -v "${BINARIES_DIR}:/binaries" \
  -v "${REPO_ROOT}:/repo" \
  reveng-compile \
  uv run python scripts/pipeline/collect_binaries.py \
    --build-root /build \
    --output-dir /binaries \
    --min-text-size 64 \
    --manifest /data/manifests/binary_manifest.csv

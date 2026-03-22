#!/bin/bash
#SBATCH --job-name=reveng-download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ichters.fe@gmail.com

set -euo pipefail
mkdir -p logs

WORK_ROOT="${WORK}/reveng-data"
SOURCES="${WORK_ROOT}/sources"
MANIFESTS="${WORK_ROOT}/manifests"

mkdir -p "${SOURCES}"/{debian,gnu,thestack} "${MANIFESTS}" "${WORK_ROOT}/logs/download"

# Download GNU sources
echo "Downloading GNU sources"
uv run python scripts/pipeline/download_gnu_sources.py \
    --output-dir "${SOURCES}/gnu" \
    2>&1 | tee "${WORK_ROOT}/logs/download/gnu.log"

# Download Debian sources
echo "Downloading Debian sources"
uv run python scripts/pipeline/download_debian_sources.py \
    --output-dir "${SOURCES}/debian" \
    --max-packages 5000 \
    2>&1 | tee "${WORK_ROOT}/logs/download/debian.log"

# Download The Stack sources
# uv run python scripts/pipeline/download_thestack.py \
#     --output-dir "${SOURCES}/thestack" \
#     --min-stars 5 \
#     --max-files 500000 \
#     2>&1 | tee "${WORK_ROOT}/logs/download/thestack.log"

# Generate package list
echo "Generating package list"
uv run python scripts/pipeline/generate_package_list.py \
    --sources-dir "${SOURCES}" \
    --output "${MANIFESTS}/package_list.txt"

echo "Package list: ${MANIFESTS}/package_list.txt"
wc -l "${MANIFESTS}/package_list.txt"
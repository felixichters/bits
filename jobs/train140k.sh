#!/bin/bash
#SBATCH --job-name=rev-eng-train
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ichters.fe@gmail.com

mkdir -p logs

uv run python -m reveng_ml train \
        --data-path data/train140k/ \
        --lr 1e-4 \
        --epochs 5 \
        --batch-size 256 \
        --model-dir models/m140k/

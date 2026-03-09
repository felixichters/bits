# Introduction

This project contains a machine learning system to identify **function boundaries** and **instruction boundaries** in stripped x86/ARM binary executable files using a BERT-based transformer model.

## How it works

The model operates directly on raw bytes of the `.text` section. A shared BERT encoder processes chunks of binary data. Two independent classification heads predict per-byte labels:

- **Function boundary head** (3 classes): `O` (other), `B-FUNC` (function start), `E-FUNC` (function end)
- **Instruction boundary head** (2 classes): `NOT-START`, `INST-START`

This multi-task dual-head architecture follows the approach of [XDA](https://arxiv.org/pdf/2010.00770).

Ground truth for function boundaries is extracted from `.symtab` / `.eh_frame` ELF sections. Ground truth for instruction boundaries is obtained by linearly disassembling the `.text` section with [Capstone](https://www.capstone-engine.org/).

## Repository Structure

-   `src/reveng_ml/`: Main Python source code for the ML pipeline
-   `data/`: Raw/processed datasets
-   `jobs/`: SLURM job scripts for cluster training
-   `scripts/`: Helper and utility scripts
-   `notebooks/`: Jupyter notebooks for experimentation, analysis, and visualization
-   `tests/`: Unit tests
-   `pyproject.toml`: Project metadata and dependencies

## Setup

You will need [uv](https://github.com/astral-sh/uv). To also enable evaluation against XDA, set up the [XDA model](https://arxiv.org/pdf/2010.00770):
```bash
./XDASetup.sh
```

## Usage

Run using `uv`. Dependencies will be installed automatically by `uv`.

```bash
# show all cli commands
uv run python -m reveng_ml --help
# or for a specific command
uv run python -m reveng_ml <command> --help
```

### 1. Split dataset

```bash
uv run python -m reveng_ml split-dataset --input-dir <raw-binaries-dir> --train-dir <train-dir> --test-dir <test-dir>
```

### 2. Create dataset

Preprocesses binaries into chunked, labelled tensors and saves them to a `.dataset` file for fast reuse.

```bash
uv run python -m reveng_ml create-dataset --input-dir <train-dir> --output-path <output>.dataset
```

### 3. Train

`--data-path` accepts either a pre-built `.dataset` file (fast) or a raw binary directory (dataset is built on the fly).

```bash
uv run python -m reveng_ml train --data-path <train>.dataset --model-dir <model-dir>
# or
uv run python -m reveng_ml train --data-path <train-dir> --model-dir <model-dir>
```

**Key training options:**

| Option | Default | Description |
|---|---|---|
| `--task` | `both` | `function`, `instruction`, or `both` |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `32` | Samples per batch |
| `--lr` | `5e-5` | Learning rate |
| `--class-weight` | auto | Weight for boundary classes (B-FUNC, E-FUNC). If unset, computed from label distribution |
| `--inst-loss-weight` | `1.0` | Weight of instruction loss relative to function loss |
| `--arch` | `x86_64` | Architecture for disassembly: `x86_64`, `x86_32`, `arm` |

### 4. Evaluate

```bash
uv run python -m reveng_ml evaluate --model-path <model>.bin --data-path <test>.dataset
# or
uv run python -m reveng_ml evaluate --model-path <model>.bin --data-path <test-dir>

# compare against xda baseline
uv run python -m reveng_ml evaluate --model-path <model>.bin --data-path <test-dir> --compare-xda
```

## Unit tests

```bash
uv run pytest
```

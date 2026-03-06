# RevEng: Function Boundary Identification

This project contains a machine learning pipeline to identify function and instruction boundaries in stripped binary executable files.

## Repository Structure

-   `src/reveng_ml/`: Main Python source code for the ML pipeline
-   `data/`: Raw/processed data sets
-   `scripts/`: Contains helper and utility scripts
-   `notebooks/`: Jupyter notebooks for experimentation, analysis, and visualization
-   `tests/`: Unit tests
-   `pyproject.toml`: Project metadata and dependencies

## Setup

To set up the development environment, you will need [uv](https://github.com/astral-sh/uv) as well as miniforge.
Then setup the [xda-model](https://arxiv.org/pdf/2010.00770) project to allow evaluation against it:
```bash
./XDASetup.sh
```

### Usage

Run using `uv`. Dependencies will be installed automatically by `uv`.
```bash
# Show CLI options
uv run python -m reveng_ml --help
```

Train using training data in `data/train`
```bash

uv run python -m reveng_ml train
# Or with custom config
uv run python -m reveng_ml train --epochs 3 --batch-size 32 --lr 0.00005 --data-path data/train --model-dir model --class-weight 100
```

Evaluate using test data in `data/test`
```bash
uv run python -m reveng_ml evaluate
# Or with custom config
uv run python -m reveng_ml evaluate --batch-size 32 --data-path data/train --model-path model/trained_model.bin
```

### Unit tests

Run unit tests using:
```bash
uv run pytest
```

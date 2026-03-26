"""
XDA baseline inference for comparative evaluation.

Loads the XDA model (a fairseq RoBERTa variant) and runs function boundary
prediction on the same chunks used by the main model.
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Directory containing the vendored XDA fairseq fork + checkpoints
XDA_DIR = Path(__file__).resolve().parent / "XDA"


def infer_xda(dataset: Dataset, xda_dir: Path | None = None) -> tuple[list[int], list[int]]:
    """
    Run XDA inference on the same dataset used by the main model.

    XDA only supports function boundary detection, so instruction labels
    are ignored. The dataset must yield 3-tuples of
    (byte_tensor, func_labels, inst_labels).

    Args:
        dataset: A BinaryChunkDataset (or compatible) yielding 3-tuples.
        xda_dir: Path to the XDA directory containing fairseq, checkpoints,
                 and data-bin. Defaults to the vendored copy next to this file.

    Returns:
        (all_preds, all_labels) — flat lists of per-byte predictions and
        ground-truth function boundary labels.
    """
    if xda_dir is None:
        xda_dir = XDA_DIR

    xda_dir = Path(xda_dir).resolve()

    import os
    original_cwd = os.getcwd()
    original_sys_path = sys.path.copy()

    try:
        sys.path.insert(0, str(xda_dir))
        os.chdir(xda_dir)

        from fairseq.models.roberta import RobertaModel  # noqa: E402 — XDA's vendored fairseq

        xda = RobertaModel.from_pretrained(
            "checkpoints/finetune_msvs_funcbound_64",
            "checkpoint_best.pt",
            "data-bin/funcbound_msvs_64",
            bpe=None,
            user_dir="finetune_tasks",
        )
        xda.eval()
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path

    print("Starting XDA inference...")
    all_preds: list[int] = []
    all_labels: list[int] = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    progress_bar = tqdm(loader, desc="XDA Inference", leave=False)

    with torch.no_grad():
        for batch_data, batch_func_labels, _batch_inst_labels in progress_bar:
            hex_str = " ".join(hex(b)[2:].ljust(2, "0") for b in batch_data[0])
            encoded_tokens = xda.encode(hex_str)

            logprobs = xda.predict("funcbound", encoded_tokens)
            predictions = logprobs.argmax(dim=2).view(-1).tolist()

            all_preds.extend(predictions)
            all_labels.extend(batch_func_labels.cpu().numpy().flatten().tolist())

    print("XDA inference complete.")
    return all_preds, all_labels

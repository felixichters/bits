
import torch
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset

from reveng_ml.model import get_model
from reveng_ml.evaluate import Evaluator


class _TinyEvalDataset(Dataset):
    """Two fixed chunks, each of length 16, for evaluator tests."""

    def __init__(self, num_chunks: int = 2, chunk_size: int = 16, num_labels: int = 3):
        self.chunks = []
        for _ in range(num_chunks):
            data = torch.randint(0, 257, (chunk_size,), dtype=torch.long)
            # Assign labels cycling through 0, 1, 2 to ensure all classes appear
            labels = torch.tensor([i % num_labels for i in range(chunk_size)], dtype=torch.long)
            self.chunks.append((data, labels))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def test_evaluator_evaluate_returns_string():
    """
    Evaluator.evaluate() returns a non-empty string containing classification report keywords.
    The model forward call is mocked to return deterministic fake logits so no GPU/training
    is needed.
    """
    chunk_size = 16
    num_labels = 3
    dataset = _TinyEvalDataset(num_chunks=2, chunk_size=chunk_size, num_labels=num_labels)
    model = get_model()

    # Build fake logits: shape (batch_size, chunk_size, num_labels).
    # We predict class 0 for every token by giving class 0 a large score.
    def fake_forward(*args, **kwargs):
        batch_input = kwargs.get("input_ids", args[0] if args else None)
        batch_size = batch_input.shape[0]
        seq_len = batch_input.shape[1]
        fake_logits = torch.zeros(batch_size, seq_len, num_labels)
        fake_logits[:, :, 0] = 10.0  # class 0 wins everywhere
        output = MagicMock()
        output.logits = fake_logits
        return output

    evaluator = Evaluator(model=model, dataset=dataset, batch_size=2, compare_xda=False)

    with patch.object(evaluator.model, "forward", side_effect=fake_forward):
        report = evaluator.evaluate()

    assert isinstance(report, str), "evaluate() must return a string"
    assert len(report) > 0, "evaluate() must return a non-empty string"

    report_lower = report.lower()
    assert "precision" in report_lower or "recall" in report_lower, (
        "Expected classification report keywords ('precision' or 'recall') in the output"
    )


import torch
import pytest
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset

from reveng_ml.model import get_model, DualHeadOutput
from reveng_ml.evaluate import Evaluator


class _TinyEvalDataset(Dataset):
    """Fixed chunks for evaluator tests, returns 3-tuples."""

    def __init__(self, num_chunks: int = 2, chunk_size: int = 16, num_func_labels: int = 3, num_inst_labels: int = 2):
        self.chunks = []
        for _ in range(num_chunks):
            data = torch.randint(0, 257, (chunk_size,), dtype=torch.long)
            func_labels = torch.tensor([i % num_func_labels for i in range(chunk_size)], dtype=torch.long)
            inst_labels = torch.tensor([i % num_inst_labels for i in range(chunk_size)], dtype=torch.long)
            self.chunks.append((data, func_labels, inst_labels))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def test_evaluator_evaluate_returns_string():
    """
    Evaluator.evaluate() returns a non-empty string containing classification report keywords.
    """
    chunk_size = 16
    dataset = _TinyEvalDataset(num_chunks=2, chunk_size=chunk_size)
    model = get_model()

    def fake_forward(*args, **kwargs):
        batch_input = kwargs.get("input_ids", args[0] if args else None)
        batch_size = batch_input.shape[0]
        seq_len = batch_input.shape[1]
        func_logits = torch.zeros(batch_size, seq_len, 3)
        func_logits[:, :, 0] = 10.0
        inst_logits = torch.zeros(batch_size, seq_len, 2)
        inst_logits[:, :, 0] = 10.0
        return DualHeadOutput(func_logits=func_logits, inst_logits=inst_logits)

    evaluator = Evaluator(model=model, dataset=dataset, batch_size=2, compare_xda=False)

    with patch.object(evaluator.model, "forward", side_effect=fake_forward):
        report = evaluator.evaluate()

    assert isinstance(report, str), "evaluate() must return a string"
    assert len(report) > 0, "evaluate() must return a non-empty string"

    report_lower = report.lower()
    assert "precision" in report_lower or "recall" in report_lower, (
        "Expected classification report keywords ('precision' or 'recall') in the output"
    )


def test_evaluator_instruction_task():
    """Evaluator with task='instruction' produces instruction boundary report."""
    chunk_size = 16
    dataset = _TinyEvalDataset(num_chunks=2, chunk_size=chunk_size)
    model = get_model(task="instruction")

    def fake_forward(*args, **kwargs):
        batch_input = kwargs.get("input_ids", args[0] if args else None)
        batch_size = batch_input.shape[0]
        seq_len = batch_input.shape[1]
        inst_logits = torch.zeros(batch_size, seq_len, 2)
        inst_logits[:, :, 0] = 10.0
        return DualHeadOutput(func_logits=None, inst_logits=inst_logits)

    evaluator = Evaluator(model=model, dataset=dataset, batch_size=2, compare_xda=False, task="instruction")

    with patch.object(evaluator.model, "forward", side_effect=fake_forward):
        report = evaluator.evaluate()

    assert isinstance(report, str)
    assert len(report) > 0
    assert "precision" in report.lower() or "recall" in report.lower()


def test_evaluator_both_task():
    """Evaluator with task='both' produces reports for both tasks."""
    chunk_size = 16
    dataset = _TinyEvalDataset(num_chunks=2, chunk_size=chunk_size)
    model = get_model(task="both")

    def fake_forward(*args, **kwargs):
        batch_input = kwargs.get("input_ids", args[0] if args else None)
        batch_size = batch_input.shape[0]
        seq_len = batch_input.shape[1]
        func_logits = torch.zeros(batch_size, seq_len, 3)
        func_logits[:, :, 0] = 10.0
        inst_logits = torch.zeros(batch_size, seq_len, 2)
        inst_logits[:, :, 0] = 10.0
        return DualHeadOutput(func_logits=func_logits, inst_logits=inst_logits)

    evaluator = Evaluator(model=model, dataset=dataset, batch_size=2, compare_xda=False, task="both")

    with patch.object(evaluator.model, "forward", side_effect=fake_forward):
        report = evaluator.evaluate()

    assert isinstance(report, str)
    # Should contain two reports joined together
    assert len(report) > 0


import torch
from torch.utils.data import Dataset
from transformers import BertForTokenClassification

from reveng_ml.model import get_model
from reveng_ml.trainer import Trainer


class _TinyDataset(Dataset):
    """Minimal in-memory dataset with controllable label counts."""

    def __init__(self, func_label_counts: torch.Tensor, inst_label_counts: torch.Tensor = None, num_chunks: int = 4, chunk_size: int = 16):
        self._func_label_counts = func_label_counts
        self._inst_label_counts = inst_label_counts if inst_label_counts is not None else torch.tensor([100, 20], dtype=torch.long)
        self.chunks = []
        for _ in range(num_chunks):
            data = torch.randint(0, 257, (chunk_size,), dtype=torch.long)
            func_labels = torch.zeros(chunk_size, dtype=torch.long)
            inst_labels = torch.zeros(chunk_size, dtype=torch.long)
            self.chunks.append((data, func_labels, inst_labels))

    def get_label_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._func_label_counts, self._inst_label_counts

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def _make_dataset(func_label_counts=None, inst_label_counts=None):
    if func_label_counts is None:
        func_label_counts = torch.tensor([100, 10, 10], dtype=torch.long)
    return _TinyDataset(func_label_counts=func_label_counts, inst_label_counts=inst_label_counts)


def test_trainer_init_manual_class_weight(tmp_path):
    """Trainer with explicit class_weight_boundary sets func class_weights to [1, w, w]."""
    dataset = _make_dataset()
    model = get_model()

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=tmp_path / "models",
        class_weight_boundary=5.0,
    )

    weights = trainer.func_class_weights.cpu()
    assert weights.shape == (3,)
    assert torch.isclose(weights[0], torch.tensor(1.0), atol=1e-5), (
        f"O weight should be 1.0, got {weights[0].item()}"
    )
    assert torch.isclose(weights[1], torch.tensor(5.0), atol=1e-5), (
        f"B-FUNC weight should be 5.0, got {weights[1].item()}"
    )
    assert torch.isclose(weights[2], torch.tensor(5.0), atol=1e-5), (
        f"E-FUNC weight should be 5.0, got {weights[2].item()}"
    )


def test_trainer_init_dynamic_class_weight(tmp_path):
    """Trainer without class_weight_boundary computes dynamic weights from the dataset."""
    dataset = _make_dataset(func_label_counts=torch.tensor([100, 10, 10], dtype=torch.long))
    model = get_model()

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=tmp_path / "models",
        class_weight_boundary=None,
    )

    weights = trainer.func_class_weights.cpu()
    assert weights.shape == (3,), f"Expected shape (3,), got {weights.shape}"
    assert (weights > 0).all(), "All dynamic weights must be positive"


def test_trainer_save_model(tmp_path):
    """trainer.save_model writes a non-empty file at the expected path."""
    dataset = _make_dataset()
    model = get_model()
    model_dir = tmp_path / "models"

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=model_dir,
        class_weight_boundary=1.0,
    )

    filename = "test.bin"
    trainer.save_model(filename)

    save_path = model_dir / filename
    assert save_path.exists(), f"Expected saved model at {save_path}"
    assert save_path.stat().st_size > 0, "Saved model file must not be empty"


def test_trainer_multitask(tmp_path):
    """Trainer with task='both' initializes both loss functions and can train."""
    dataset = _make_dataset()
    model = get_model(task="both")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=tmp_path / "models",
        class_weight_boundary=5.0,
        task="both",
    )

    assert hasattr(trainer, 'func_loss_fct')
    assert hasattr(trainer, 'inst_loss_fct')

    # Should be able to run one epoch without error
    trainer.train(epochs=1)


def test_trainer_instruction_only(tmp_path):
    """Trainer with task='instruction' only sets up instruction loss."""
    dataset = _make_dataset()
    model = get_model(task="instruction")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=tmp_path / "models",
        task="instruction",
    )

    assert hasattr(trainer, 'inst_loss_fct')
    assert not hasattr(trainer, 'func_loss_fct')

    trainer.train(epochs=1)

def test_trainer_function_only(tmp_path):
    """Trainer with task='function' uses BertForTokenClassification model and only sets up function loss."""
    dataset = _make_dataset()
    model = get_model(task="function")

    assert isinstance(model, BertForTokenClassification)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        batch_size=2,
        model_dir=tmp_path / "models",
        task="function",
    )

    assert not hasattr(trainer, 'inst_loss_fct')
    assert hasattr(trainer, 'func_loss_fct')

    trainer.train(epochs=1)

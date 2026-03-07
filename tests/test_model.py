
import torch
import pytest
from transformers import BertForTokenClassification

from reveng_ml.model import get_model


def test_get_model_default():
    """get_model() with defaults returns a BertForTokenClassification with expected config."""
    model = get_model()

    assert isinstance(model, BertForTokenClassification)
    assert model.config.num_labels == 3
    assert model.config.vocab_size == 257
    assert model.config.hidden_size == 256


def test_get_model_custom():
    """get_model with custom parameters reflects those values in the config."""
    model = get_model(num_labels=5, hidden_size=128, num_attention_heads=4, num_hidden_layers=2)

    assert isinstance(model, BertForTokenClassification)
    assert model.config.num_labels == 5
    assert model.config.hidden_size == 128
    assert model.config.num_attention_heads == 4
    assert model.config.num_hidden_layers == 2


def test_model_forward_pass():
    """A forward pass with a small (2, 16) input produces logits of shape (2, 16, 3)."""
    model = get_model()
    model.eval()

    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.logits.shape == (2, 16, 3), (
        f"Expected logits shape (2, 16, 3), got {outputs.logits.shape}"
    )

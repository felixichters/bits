
import torch
import pytest
from transformers import BertForTokenClassification

from reveng_ml.model import get_model, DualHeadBertForTokenClassification, DualHeadOutput


def test_get_model_default():
    """get_model() with defaults returns a DualHeadBertForTokenClassification (task='both')."""
    model = get_model()

    assert isinstance(model, DualHeadBertForTokenClassification)
    assert model.config.vocab_size == 257
    assert model.config.hidden_size == 256
    assert model.num_func_labels == 3
    assert model.num_inst_labels == 2


def test_get_model_function_only():
    """get_model(task='function') returns a BertForTokenClassification."""
    model = get_model(task="function")

    assert isinstance(model, BertForTokenClassification)
    assert model.config.num_labels == 3
    assert model.config.vocab_size == 257
    assert model.config.hidden_size == 256


def test_get_model_custom():
    """get_model with custom parameters reflects those values in the config."""
    model = get_model(num_func_labels=5, hidden_size=128, num_attention_heads=4, num_hidden_layers=2)

    assert isinstance(model, DualHeadBertForTokenClassification)
    assert model.config.hidden_size == 128
    assert model.config.num_attention_heads == 4
    assert model.config.num_hidden_layers == 2
    assert model.num_func_labels == 5


def test_model_forward_pass():
    """A forward pass with a small (2, 16) input produces func and inst logits."""
    model = get_model()
    model.eval()

    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    assert outputs.func_logits.shape == (2, 16, 3)
    assert outputs.inst_logits.shape == (2, 16, 2)


def test_get_model_dual_head():
    """get_model(task='both') returns DualHeadBertForTokenClassification."""
    model = get_model(task="both")
    assert isinstance(model, DualHeadBertForTokenClassification)


def test_get_model_instruction_only():
    """get_model(task='instruction') returns DualHeadBertForTokenClassification."""
    model = get_model(task="instruction")
    assert isinstance(model, DualHeadBertForTokenClassification)


def test_dual_head_forward_both():
    """Dual-head model with task='both' produces both func and inst logits."""
    model = get_model(task="both")
    model.eval()
    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, task="both")

    assert output.func_logits.shape == (2, 16, 3)
    assert output.inst_logits.shape == (2, 16, 2)


def test_dual_head_forward_function_only():
    """Dual-head model with task='function' only produces func logits."""
    model = get_model(task="both")
    model.eval()
    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, task="function")

    assert output.func_logits is not None
    assert output.inst_logits is None


def test_dual_head_forward_instruction_only():
    """Dual-head model with task='instruction' only produces inst logits."""
    model = get_model(task="both")
    model.eval()
    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, task="instruction")

    assert output.func_logits is None
    assert output.inst_logits is not None
    assert output.inst_logits.shape == (2, 16, 2)


def test_dual_head_backward_compat_logits_property():
    """DualHeadOutput.logits returns func_logits for backward compatibility."""
    model = get_model(task="both")
    model.eval()
    input_ids = torch.randint(0, 257, (2, 16), dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, task="both")

    assert torch.equal(output.logits, output.func_logits)

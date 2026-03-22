"""
Model definition for the RevEng-ML project.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertForTokenClassification


@dataclass
class DualHeadOutput:
    """Output container for the dual-head model."""

    func_logits: Optional[torch.Tensor]  # (batch, seq_len, num_func_labels)
    inst_logits: Optional[torch.Tensor]  # (batch, seq_len, num_inst_labels)

    @property
    def logits(self):
        """Backward compatibility: returns func_logits."""
        return self.func_logits


class DualHeadBertForTokenClassification(nn.Module):
    """BERT encoder with two independent classification heads for function and instruction boundaries."""

    def __init__(self, config: BertConfig, num_func_labels: int = 3, num_inst_labels: int = 2):
        """Initializes the dual-head BERT encore for token classification."""
        super().__init__()
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.func_classifier = nn.Linear(config.hidden_size, num_func_labels)
        self.inst_classifier = nn.Linear(config.hidden_size, num_inst_labels)
        self.num_func_labels = num_func_labels
        self.num_inst_labels = num_inst_labels

    def forward(self, input_ids, attention_mask=None, task="both"):
        """
        Performs a forward pass through the model, classifying function and/or
        instruction boundaries based on the specified task.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)

        func_logits = None
        inst_logits = None

        if task in ("function", "both"):
            func_logits = self.func_classifier(sequence_output)
        if task in ("instruction", "both"):
            inst_logits = self.inst_classifier(sequence_output)

        return DualHeadOutput(func_logits=func_logits, inst_logits=inst_logits)


def get_model(
    vocab_size: int = 257,
    num_func_labels: int = 3,
    num_inst_labels: int = 2,
    hidden_size: int = 256,
    num_attention_heads: int = 8,
    num_hidden_layers: int = 4,
    intermediate_size: int = 1024,
    task: str = "both",
) -> nn.Module:
    """
    Initializes a new BERT model for token classification with a custom configuration.

    Args:
        vocab_size (int): Size of vocabulary 256 bytes (+ special token)
        num_func_labels (int): Number of function boundary labels
        num_inst_labels (int): Number of instruction boundary labels
        hidden_size (int): Dimensionality of the model's hidden layers
        num_attention_heads (int): Number of attention heads in the transformer
        num_hidden_layers (int): Number of transformer layers
        intermediate_size (int): The size of the feed-forward layer in the transformer
        task (str): "function" (backward-compatible), "instruction", or "both"

    Returns:
        A model instance.
    """
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        max_position_embeddings=512,
        type_vocab_size=1,
    )

    if task == "function":
        config.num_labels = num_func_labels
        return BertForTokenClassification(config)
    else:
        return DualHeadBertForTokenClassification(
            config,
            num_func_labels=num_func_labels,
            num_inst_labels=num_inst_labels,
        )

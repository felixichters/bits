"""
Training script for the RevEng-ML project.
"""
import time
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup, BertForTokenClassification
from reveng_ml.utils import get_pytorch_device
from torch.nn.utils import clip_grad_norm_


class Trainer:
    """Trains a model using a dataset"""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        learning_rate: float = 5e-5,
        batch_size: int = 32,
        model_dir: Path = Path('./models'),
        class_weight_boundary: Optional[float] = None,
        task: str = "both",
        inst_loss_weight: float = 1.0,
    ):
        """
        Create a new Trainer class.

        Args:
            model: PyTorch model
            dataset: PyTorch dataset
            learning_rate (float): Optimizer learning rate
            batch_size (int): Samples per batch
            model_dir (Path): Model output directory
            class_weight_boundary (float | None): Weight for boundary classes (B-FUNC, E-FUNC).
                If None, weights are computed dynamically from the dataset using inverse frequency.
            task (str): "function", "instruction", or "both"
            inst_loss_weight (float): Weight for instruction loss relative to function loss in multi-task mode
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)
        self.task = task
        self.inst_loss_weight = inst_loss_weight

        func_counts, inst_counts = dataset.get_label_counts()

        # Function boundary loss
        if task in ("function", "both"):
            if class_weight_boundary is not None:
                self.func_class_weights = torch.tensor([1.0, class_weight_boundary, class_weight_boundary]).to(self.device)
                print(f"Using manual function class weights: {self.func_class_weights.tolist()}")
            else:
                total = func_counts.sum().float()
                self.func_class_weights = (total / (3 * func_counts.float())).to(self.device)
                print(f"Label distribution: O={func_counts[0]:,}, B-FUNC={func_counts[1]:,}, E-FUNC={func_counts[2]:,}")
                print(f"Using dynamic function class weights: {[f'{w:.2f}' for w in self.func_class_weights.tolist()]}")
            self.func_loss_fct = torch.nn.CrossEntropyLoss(weight=self.func_class_weights)

        # Instruction boundary loss
        if task in ("instruction", "both"):
            total = inst_counts.sum().float()
            self.inst_class_weights = (total / (2 * inst_counts.float())).to(self.device)
            print(f"Instruction label distribution: NOT-START={inst_counts[0]:,}, INST-START={inst_counts[1]:,}")
            print(f"Using dynamic instruction class weights: {[f'{w:.2f}' for w in self.inst_class_weights.tolist()]}")
            self.inst_loss_fct = torch.nn.CrossEntropyLoss(weight=self.inst_class_weights)

    def train(self, epochs: int = 3):
        """
        Train for a number of epochs.

        Args:
            epochs (int): Epoch count
        """
        self.model.train()

        total_steps = len(self.loader) * epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        print(f"LR schedule: linear warmup for {warmup_steps} steps, linear decay over {total_steps} total steps.")

        for epoch in range(epochs):
            print(f"--- Starting Epoch {epoch + 1}/{epochs} ---")
            epoch_start_time = time.time()
            total_loss = 0

            # Wrap with tqdm() to show progress_bar
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for i, (batch_data, batch_func_labels, batch_inst_labels) in enumerate(progress_bar):

                batch_data = batch_data.to(self.device)
                batch_func_labels = batch_func_labels.to(self.device)
                batch_inst_labels = batch_inst_labels.to(self.device)

                # Clear prev. gradients
                self.model.zero_grad()

                # Backward-compatible path for function-only with original model
                if self.task == "function" and isinstance(self.model, BertForTokenClassification):
                    outputs = self.model(input_ids=batch_data)
                    logits = outputs.logits
                    loss = self.func_loss_fct(logits.view(-1, 3), batch_func_labels.view(-1))
                else:
                    outputs = self.model(input_ids=batch_data, task=self.task)
                    loss = torch.tensor(0.0, device=self.device)

                    if self.task in ("function", "both"):
                        func_loss = self.func_loss_fct(
                            outputs.func_logits.view(-1, 3), batch_func_labels.view(-1)
                        )
                        loss = loss + func_loss

                    if self.task in ("instruction", "both"):
                        inst_loss = self.inst_loss_fct(
                            outputs.inst_logits.view(-1, 2), batch_inst_labels.view(-1)
                        )
                        loss = loss + self.inst_loss_weight * inst_loss

                total_loss += loss.item()

                # Backward pass and optimization
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()

                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

            avg_loss = total_loss / len(self.loader)
            epoch_time = time.time() - epoch_start_time
            self.save_model("reveng_model_epoch" + str(epoch+1) + ".bin")
            print(f"--- Epoch {epoch + 1} Summary ---")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch Time: {epoch_time:.2f} seconds")
            print("-" * (25 + len(str(epoch+1))))


    def save_model(self, filename: str = "reveng_model.bin"):
        """Saves the model state"""
        os.makedirs(self.model_dir, exist_ok=True)
        save_path = self.model_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

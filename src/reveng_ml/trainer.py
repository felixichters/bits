"""
Training script for the RevEng-ML project.
"""
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup
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
        class_weight_boundary: float | None = None
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
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True)

        if class_weight_boundary is not None:
            self.class_weights = torch.tensor([1.0, class_weight_boundary, class_weight_boundary]).to(self.device)
            print(f"Using manual class weights: {self.class_weights.tolist()}")
        else:
            counts = dataset.get_label_counts()
            total = counts.sum().float()
            num_classes = 3
            self.class_weights = (total / (num_classes * counts.float())).to(self.device)
            print(f"Label distribution: O={counts[0]:,}, B-FUNC={counts[1]:,}, E-FUNC={counts[2]:,}")
            print(f"Using dynamic class weights: {[f'{w:.2f}' for w in self.class_weights.tolist()]}")

        self.loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)

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
            
            for i, (batch_data, batch_labels) in enumerate(progress_bar):
                
                #no attention mask needed since currently all inputs are completely filled and not padded
                #att_msk = torch.zeros(batch_data.shape[0],512,dtype=batch_data.dtype)
                #att_msk[:batch_data.size(0),:batch_data.size(1)] = torch.ones_like(batch_data)
                #att_msk = att_msk.to(self.device)

                batch_labels = batch_labels.to(self.device)
                batch_data = batch_data.to(self.device)

                # Clear prev. gradients
                self.model.zero_grad()

                # Forward pass
                outputs = self.model(input_ids=batch_data)
                
                # Compute loss with class weights
                logits = outputs.logits
                loss = self.loss_fct(logits.view(-1, 3), batch_labels.view(-1))

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

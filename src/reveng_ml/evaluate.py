"""
Evaluation script for the RevEng-ML project.
"""
import os.path
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pickle
import subprocess
from transformers import BertForTokenClassification

from reveng_ml.utils import get_pytorch_device

import os
from pathlib import Path

class Evaluator:
    """Evaluates a trained model"""

    def __init__(self, model: torch.nn.Module, dataset: Dataset, batch_size: int = 32, compare_xda: bool = False, task: str = "both"):
        """
        Creates a new Evaluator class

        Args:
            model: Trained PyTorch model to evaluate
            dataset: PyTorch dataset
            batch_size (int): Batch size for evaluation
            compare_xda (bool): Run XDA baseline comparison (disabled by default)
            task (str): "function", "instruction", or "both"
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.compare_xda = compare_xda
        self.task = task
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self) -> str:
        """
        Execute evaluation

        Returns:
            A string containing the classification report(s) from scikit-learn
        """
        self.model.eval()
        func_preds = []
        func_labels = []
        inst_preds = []
        inst_labels = []

        print("Starting evaluation...")
        progress_bar = tqdm(self.loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch_data, batch_func_labels, batch_inst_labels in progress_bar:
                batch_data = batch_data.to(self.device)

                # Backward-compatible path for function-only with original model
                if self.task == "function" and isinstance(self.model, BertForTokenClassification):
                    outputs = self.model(input_ids=batch_data)
                    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy().flatten()
                    func_preds.extend(predictions)
                    func_labels.extend(batch_func_labels.cpu().numpy().flatten())
                else:
                    outputs = self.model(input_ids=batch_data, task=self.task)

                    if self.task in ("function", "both"):
                        predictions = torch.argmax(outputs.func_logits, dim=-1).cpu().numpy().flatten()
                        func_preds.extend(predictions)
                        func_labels.extend(batch_func_labels.cpu().numpy().flatten())

                    if self.task in ("instruction", "both"):
                        predictions = torch.argmax(outputs.inst_logits, dim=-1).cpu().numpy().flatten()
                        inst_preds.extend(predictions)
                        inst_labels.extend(batch_inst_labels.cpu().numpy().flatten())

        print("Evaluation complete.")

        reports = []

        if self.compare_xda:
            print("Starting xda evaluation...")
            xdaDatasetInfoPath = os.path.abspath(Path("src/reveng_ml/ComparativeEvaluation/XDA/dataset.info"))
            xdaResultPath = os.path.abspath(Path("src/reveng_ml/ComparativeEvaluation/XDA/result.inferred"))
            xdaExecutablePath = os.path.abspath(Path("src/reveng_ml/ComparativeEvaluation/runInferXDA.sh"))
            with open(xdaDatasetInfoPath,"wb") as f:
                pickle.dump([os.path.abspath(self.dataset.data_path),self.dataset.chunk_size,self.dataset.stride],f,0)

            try:
                subprocessResult=subprocess.run(["./src/reveng_ml/ComparativeEvaluation/runInferXDA.sh", str(xdaDatasetInfoPath),str(xdaResultPath)],shell=False,check=True,capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"Error using XDA to infer the dataset {xdaResultPath}: {e.stderr.decode().strip()}")
                raise

            with open(xdaResultPath,"rb") as f:
                xdaResult = pickle.load(f)
                xda_all_labels = xdaResult[0]
                xda_all_preds = xdaResult[1]

            report_xda = classification_report(
                xda_all_labels,
                xda_all_preds,
                target_names=['O', 'B-FUNC', 'E-FUNC'],
                zero_division=0
                )

        # Function boundary report
        if func_preds:
            report = classification_report(
                func_labels,
                func_preds,
                target_names=['O', 'B-FUNC', 'E-FUNC'],
                zero_division=0
            )

            total = len(func_labels)
            o_count = sum(1 for l in func_labels if l == 0)
            b_count = sum(1 for l in func_labels if l == 1)
            e_count = sum(1 for l in func_labels if l == 2)

            print("\n--- Function Boundary Classification Report ---")
            print(f"Class Distribution:")
            print(f"  O (non-boundary): {o_count:,} ({100*o_count/total:.2f}%)")
            print(f"  B-FUNC: {b_count:,} ({100*b_count/total:.2f}%)")
            print(f"  E-FUNC: {e_count:,} ({100*e_count/total:.2f}%)")
            print(f"\n{report}")

            cm = confusion_matrix(func_labels, func_preds, labels=[0, 1, 2])
            print("Confusion Matrix:")
            print("              Predicted")
            print("              O      B-FUNC  E-FUNC")
            print(f"Actual O      {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
            print(f"       B-FUNC {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
            print(f"       E-FUNC {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")
            print("-----------------------------\n")
            reports.append(report)

        # Instruction boundary report
        if inst_preds:
            report = classification_report(
                inst_labels,
                inst_preds,
                target_names=['NOT-START', 'INST-START'],
                zero_division=0
            )

            total = len(inst_labels)
            ns_count = sum(1 for l in inst_labels if l == 0)
            is_count = sum(1 for l in inst_labels if l == 1)

            print("\n--- Instruction Boundary Classification Report ---")
            print(f"Class Distribution:")
            print(f"  NOT-START: {ns_count:,} ({100*ns_count/total:.2f}%)")
            print(f"  INST-START: {is_count:,} ({100*is_count/total:.2f}%)")
            print(f"\n{report}")

            cm = confusion_matrix(inst_labels, inst_preds, labels=[0, 1])
            print("Confusion Matrix:")
            print("              Predicted")
            print("              NOT-START  INST-START")
            print(f"Actual NOT-START  {cm[0][0]:<10} {cm[0][1]:<10}")
            print(f"       INST-START {cm[1][0]:<10} {cm[1][1]:<10}")
            print("-----------------------------\n")
            reports.append(report)

        if self.compare_xda:
            print("\n--- Classification Report XDA ---")
            print(report_xda)
            print("-----------------------------\n")

        return "\n".join(reports)

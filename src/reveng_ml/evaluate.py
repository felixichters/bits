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

from reveng_ml.utils import get_pytorch_device

import os
from pathlib import Path

class Evaluator:
    """Evaluates a trained model"""

    def __init__(self, model: torch.nn.Module, dataset: Dataset, batch_size: int = 32):
        """
        Creates a new Evaluator class

        Args:
            model: Trained PyTorch model to evaluate
            dataset: PyTorch dataset
            batch_size (int): Batch size for evaluation
        """
        self.device = get_pytorch_device()
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def evaluate(self) -> str:
        """
        Execute evaluation

        Returns:
            A string containing the classification report from scikit-learn
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        print("Starting evaluation...")
        progress_bar = tqdm(self.loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for batch_data, batch_labels in progress_bar:
                batch_data = batch_data.to(self.device)
                
                # Get model predictions
                outputs = self.model(input_ids=batch_data)
                logits = outputs.logits
                
                # class with the highest score
                predictions = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
                
                all_preds.extend(predictions)
                all_labels.extend(batch_labels.cpu().numpy().flatten())

        print("Evaluation complete.")

        
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
        

        # Print a classification report
        report = classification_report(
            all_labels,
            all_preds,
            # O = None, B-FUNC = Beginning of a function, E-FUNC = End of a function
            target_names=['O', 'B-FUNC', 'E-FUNC'],
            zero_division=0
        )

        # Calculate class distribution
        total = len(all_labels)
        o_count = sum(1 for l in all_labels if l == 0)
        b_count = sum(1 for l in all_labels if l == 1)
        e_count = sum(1 for l in all_labels if l == 2)

        print("\n--- Classification Report own model ---")
        print(f"Class Distribution:")
        print(f"  O (non-boundary): {o_count:,} ({100*o_count/total:.2f}%)")
        print(f"  B-FUNC: {b_count:,} ({100*b_count/total:.2f}%)")
        print(f"  E-FUNC: {e_count:,} ({100*e_count/total:.2f}%)")
        print(f"\n{report}")

        # Show confusion matrix for boundary classes
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print("              Predicted")
        print("              O      B-FUNC  E-FUNC")
        print(f"Actual O      {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
        print(f"       B-FUNC {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
        print(f"       E-FUNC {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")
        print("-----------------------------\n")

        print("\n--- Classification Report XDA ---")
        print(report_xda)
        print("-----------------------------\n")

        return report

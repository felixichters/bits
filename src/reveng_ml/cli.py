"""
Command-line interface
"""
import typer
from pathlib import Path
import torch

from reveng_ml.evaluate import Evaluator
from reveng_ml.data import BinaryChunkDataset
from reveng_ml.model import get_model
from reveng_ml.trainer import Trainer

app = typer.Typer(help="Function boundary detection model training & evaluation")

@app.command()
def create_dataset(
    data_path: Path = typer.Option("data/train/default", "--input-dir", "-d" , help="Input Directory; all files inside will be included in the resulting dataset file"),
    chunk_size: int = typer.Option(510, "--chunk-size", "-c" , help="Size Of Chunk to be fed into model at a time"),
    stride: int = typer.Option(255, "--stride", "-s", help="Amount of stride (overlap with previous and following chunk)"),
    onlyDotText: bool = typer.Option(True, "--only-text", "-t", help="Wether the whole binary or only the .text section get used"),
    result_path: Path = typer.Option("data/default.dataset", "--output-path", "-o", help="Resulting dataset file path")
):
    """
    Create new dataset file from executabes inside given directory
    """
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"Error: Training data directory '{data_path}' is empty or does not exist.")
        raise typer.Exit(code=1)
    
    # Load training data
    print(f"Loading data from {data_path}...")
    dataset = BinaryChunkDataset(data_path=data_path, chunk_size=chunk_size, stride=stride, onlyIncludeCodeSegment=onlyDotText)
    if not dataset:
        print("Warning: The dataset is empty.")
        raise typer.Exit()

    dataset.save(result_path)
    
    
    print(f"Created dataset with {len(dataset)} chunks, {chunk_size} chunk size and {stride} stride.")
    print(f"Saved dataset to: {result_path}")


@app.command()
def train(
    data_path: Path = typer.Option("data/train/default.dataset", "--data-path", "-d", help="Training data input directory or dataset file-path"),
    model_dir: Path = typer.Option("models/default", "--model-dir", "-o", help="Model output directory"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", "-l", help="Learning rate"),
    chunk_size: int = typer.Option(510, help="Size of each binary chunk"),
    stride: int = typer.Option(255, help="Stride for overlapping chunks"),
    class_weight_boundary: float = typer.Option(100.0, "--class-weight", "-w", help="Weight for boundary classes (B-FUNC, E-FUNC). Higher = more focus on boundaries"),
):
    """
    Train a new function boundary detection model.
    """
    

    # Load training data
    print(f"Loading data from {data_path}...")
    dataset = BinaryChunkDataset(data_path=data_path, chunk_size=chunk_size, stride=stride)
    if not dataset:
        print("Warning: The dataset is empty. No training will be performed.")
        raise typer.Exit()
    print(f"Created dataset with {len(dataset)} chunks.")

    print("Initializing model...")
    model = get_model()
    # Train
    trainer = Trainer(model, dataset, learning_rate=learning_rate, batch_size=batch_size, model_dir=model_dir, class_weight_boundary=class_weight_boundary)
    trainer.train(epochs=epochs)

    # Save
    model_name = "reveng_boundary_detector_final.bin"
    trainer.save_model(model_name)
    print(f"Training complete. Model saved to {model_dir / model_name}.")

@app.command()
def evaluate(
    model_path: Path = typer.Option("models/default/reveng_boundary_detector_final.bin", "--model-path", "-m", help="Trained model path"),
    data_path: Path = typer.Option("data/test/default.dataset", "--data-path", "-d", help="Test data directory or test dataset file-path"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Evaluation batch size"),
    chunk_size: int = typer.Option(510, help="Size of each binary chunk"),
    stride: int = typer.Option(255, help="Stride for overlapping chunks"),
):
    """
    Evaluate a trained model on a test dataset.
    """
    print(f"Starting evaluation process...")

    if not model_path.exists():
        print(f"Error: Model file not found at '{model_path}'.")
        raise typer.Exit(code=1)

    """
    if data_path.is_file():
        print(f"Loading test data from dataset file {data_path}")
        dataset = BinaryChunkDataset(dataset_path=dataset_path)
    else:
        print(f"Loading test data from directory {data_path}")
        if not data_path.exists() or not any(data_path.iterdir()):
            print(f"Error: Test data directory '{data_path}' is empty or does not exist.")
            raise typer.Exit(code=1)
        """
    
    dataset = BinaryChunkDataset(data_path=data_path, chunk_size=chunk_size, stride=stride, randomizeFileOrder=False, for_evaluation=True)
    if not dataset:
        print("Warning: The test dataset is empty. No evaluation will be performed.")
        raise typer.Exit()
    print(f"Created test dataset with {len(dataset)} chunks.")

    # Load trained model
    print(f"Loading model from {model_path}...")
    model = get_model()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    # Evaluate
    evaluator = Evaluator(model, dataset, batch_size=batch_size)
    evaluator.evaluate()
    print(f"Evaluation complete.")


if __name__ == "__main__":

    app()


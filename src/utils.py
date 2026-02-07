import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args():
    """
    Parse arguments given to the script.

    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Train postpartum depression classification model using Hugging Face Trainer.")

    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        metavar="N",
        help="number of total epochs to run (default: 5)",
    )
    parser.add_argument(
        "--batch",
        default=16,
        type=int,
        metavar="N",
        help="batch size per device (default: 16)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="enable Weights & Biases logging",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="name for this training run (for wandb)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity (username or team)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="postpartum-depression-classification",
        help="wandb project name",
    )
    args = parser.parse_args()
    return args


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels

    Returns:
        dict: Dictionary containing accuracy, f1_macro, f1_weighted, precision, and recall
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }

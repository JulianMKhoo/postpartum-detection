"""
Result Analysis Script
Generate confusion matrix and detailed classification report from the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from dataset import PostpartumDataset
from transformers import DataCollatorWithPadding

# Class labels
CLASS_NAMES = [
    "Normal Pregnancy (Label 0)",
    "General Depression (Label 1)",
    "Postpartum Depression (Label 2)"
]

def load_best_model():
    """Load the best trained model from ./best_model/"""
    print("Loading best model from ./best_model/...")
    tokenizer = AutoTokenizer.from_pretrained("./best_model")
    model = AutoModelForSequenceClassification.from_pretrained("./best_model", num_labels=3).to("cuda")
    # model = PeftModel.from_pretrained(model, "./best_model")
    model.eval()  # Set to evaluation mode
    return tokenizer, model

def get_predictions(model, tokenizer, test_dataset, batch_size=256):
    """Get predictions from the model on test dataset with probabilities"""
    print(f"Generating predictions on {len(test_dataset)} test samples...")

    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    all_predictions = []
    all_labels = []
    all_probabilities = []  # Store probabilities for each sample

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_id = batch['input_ids'].to("cuda")
            att_mask = batch['attention_mask'].to("cuda")
            # Get model outputs
            outputs = model(
                input_ids=input_id,
                attention_mask=att_mask
            )

            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get predictions (argmax of probabilities)
            predictions = torch.argmax(probabilities, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            if i % 100 == 0:
                print(i, "step")

    return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot and save confusion matrix"""
    print("Generating confusion matrix...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot with seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix - Postpartum Depression Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")

    # Show plot

    return cm

def print_classification_report(y_true, y_pred):
    """Print detailed classification report"""
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print(report)

    # Also print as dictionary for more details
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    )

    return report_dict

def analyze_confidence(y_prob, y_pred, y_true, confidence_threshold=0.2):
    """
    Analyze model confidence and identify uncertain predictions.

    Args:
        y_prob: Probability distributions (N x 3)
        y_pred: Predicted labels
        y_true: True labels
        confidence_threshold: Threshold for second-highest probability (default 0.20)
                             If second-highest prob > threshold, flag as uncertain
    """
    print("\n" + "="*80)
    print("CONFIDENCE ANALYSIS")
    print("="*80)

    # Get max probability (predicted class) and second max probability
    sorted_probs = np.sort(y_prob, axis=1)  # Sort probabilities
    max_prob = sorted_probs[:, -1]  # Highest probability
    second_max_prob = sorted_probs[:, -2]  # Second-highest probability

    # Calculate confidence gap (difference between top 2 predictions)
    confidence_gap = max_prob - second_max_prob

    # Identify uncertain predictions (small gap = ambiguous)
    uncertain_mask = second_max_prob > confidence_threshold
    uncertain_indices = np.where(uncertain_mask)[0]

    print(f"\nConfidence threshold: {confidence_threshold*100:.0f}% (for second-highest probability)")
    print(f"Total samples: {len(y_prob)}")
    print(f"High confidence predictions: {np.sum(~uncertain_mask)} ({100*np.sum(~uncertain_mask)/len(y_prob):.2f}%)")
    print(f"Uncertain predictions (REFER TO DOCTOR): {len(uncertain_indices)} ({100*len(uncertain_indices)/len(y_prob):.2f}%)")

    # Statistics on confidence gap
    print(f"\nConfidence Gap Statistics:")
    print(f"  Mean gap: {np.mean(confidence_gap)*100:.2f}%")
    print(f"  Median gap: {np.median(confidence_gap)*100:.2f}%")
    print(f"  Min gap: {np.min(confidence_gap)*100:.2f}%")
    print(f"  Max gap: {np.max(confidence_gap)*100:.2f}%")

    # Among uncertain predictions, how many were correct?
    if len(uncertain_indices) > 0:
        uncertain_correct = np.sum(y_pred[uncertain_indices] == y_true[uncertain_indices])
        print(f"\nAmong uncertain predictions:")
        print(f"  Correct: {uncertain_correct}/{len(uncertain_indices)} ({100*uncertain_correct/len(uncertain_indices):.2f}%)")
        print(f"  Incorrect: {len(uncertain_indices)-uncertain_correct}/{len(uncertain_indices)} ({100*(len(uncertain_indices)-uncertain_correct)/len(uncertain_indices):.2f}%)")

    # Show some examples of uncertain predictions
    if len(uncertain_indices) > 0:
        print("\n--- Example Uncertain Predictions (first 5) ---")
        print("These cases should be REFERRED TO DOCTOR for review:")
        for i, idx in enumerate(uncertain_indices[:5]):
            print(f"\nUncertain Example {i+1}:")
            print(f"  True Label: {CLASS_NAMES[y_true[idx]]}")
            print(f"  Predicted: {CLASS_NAMES[y_pred[idx]]} {'✓ CORRECT' if y_pred[idx] == y_true[idx] else '✗ WRONG'}")
            print(f"  Confidence Gap: {confidence_gap[idx]*100:.2f}%")
            print(f"  Probabilities:")
            for class_idx in range(3):
                marker = " ← PREDICTED" if class_idx == y_pred[idx] else ""
                print(f"    {CLASS_NAMES[class_idx]}: {y_prob[idx][class_idx]*100:.2f}%{marker}")

    return uncertain_indices, confidence_gap

def analyze_errors(y_true, y_pred, y_prob, test_data):
    """Analyze misclassified samples with probabilities"""
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)

    # Find misclassified indices
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    print(f"\nTotal misclassifications: {len(error_indices)} out of {len(y_true)} ({100*len(error_indices)/len(y_true):.2f}%)")

    if len(error_indices) > 0:
        print("\nMisclassification breakdown:")
        for true_label in range(3):
            for pred_label in range(3):
                if true_label != pred_label:
                    count = np.sum((y_true == true_label) & (y_pred == pred_label))
                    if count > 0:
                        print(f"  True: {CLASS_NAMES[true_label]} → Predicted: {CLASS_NAMES[pred_label]}: {count} samples")

        # Show a few example errors with probabilities
        print("\n--- Example Misclassifications (first 5) with Probabilities ---")
        for i, idx in enumerate(error_indices[:5]):
            print(f"\nExample {i+1}:")
            print(f"  True Label: {CLASS_NAMES[y_true[idx]]}")
            print(f"  Predicted: {CLASS_NAMES[y_pred[idx]]}")
            print(f"  Probabilities:")
            for class_idx in range(3):
                marker = " ← PREDICTED" if class_idx == y_pred[idx] else (" ← TRUE" if class_idx == y_true[idx] else "")
                print(f"    {CLASS_NAMES[class_idx]}: {y_prob[idx][class_idx]*100:.2f}%{marker}")
            print(f"  Text (first 200 chars): {test_data.iloc[idx]['selftext'][:200]}...")

def main(confidence_threshold=0.20):
    """
    Main function to run all analysis

    Args:
        confidence_threshold: Threshold for flagging uncertain predictions (default 0.20)
                             Samples with second-highest probability > threshold will be flagged
    """
    print("="*80)
    print("POSTPARTUM DEPRESSION CLASSIFICATION - RESULT ANALYSIS")
    print("="*80)

    # Load best model
    tokenizer, model = load_best_model()

    # Load test dataset
    print("\nLoading test dataset...")
    test_data = pd.read_parquet("datasets/test.parquet")
    test_dataset = PostpartumDataset(test_data, tokenizer)
    print(f"Test dataset size: {len(test_dataset)} samples")

    # Get predictions with probabilities
    y_true, y_pred, y_prob = get_predictions(model, tokenizer, test_dataset)

    # Generate and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png")

    # Print detailed classification report
    print_classification_report(y_true, y_pred)

    # Analyze model confidence (IMPORTANT for clinical use!)
    uncertain_indices, confidence_gap = analyze_confidence(
        y_prob, y_pred, y_true,
        confidence_threshold=confidence_threshold
    )

    # Analyze errors with probabilities
    analyze_errors(y_true, y_pred, y_prob, test_data)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nNote: {len(uncertain_indices)} samples flagged for doctor review (confidence threshold: {confidence_threshold*100:.0f}%)")

if __name__ == "__main__":
    main()

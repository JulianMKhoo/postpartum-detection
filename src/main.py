from dataset import get_datasets
from model import deberta_model, deberta_tokenizer
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from utils import parse_args, compute_metrics


def train(args):
    # Load datasets
    train_dataset, val_dataset, test_dataset = get_datasets(deberta_tokenizer)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=deberta_model)

    # Training arguments for CPU
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        max_grad_norm=0.9,
        report_to="none",
        run_name=args.run_name if args.run_name else None,
        fp16=True,
        use_cpu=False
    )

    print(f"Training with batch_size={args.batch} on CPU")

    # Initialize Trainer
    trainer = Trainer(
        model=deberta_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate on test set
    print("\n--- Evaluating on Test Set ---")
    test_results = trainer.evaluate(test_dataset)

    print("\n--- Test Results ---")
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test F1-Score (Macro): {test_results['eval_f1_macro']:.4f}")
    print(f"Test F1-Score (Weighted): {test_results['eval_f1_weighted']:.4f}")

    # Save the best model
    print("\nSaving best model...")
    trainer.save_model("./best_model")
    debert_tokenizer.save_pretrained("./best_model")

    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    train(args)

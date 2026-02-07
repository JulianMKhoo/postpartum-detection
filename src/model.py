import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Disable MPS completely

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# lora_config = LoraConfig(
#     r=16,             # Rank (Size of the small matrices)
#     lora_alpha=32,    # Scaling factor
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.SEQ_CLS # Sequence Classification (for BERT)
# )

# Force CPU to avoid MPS OOM issues
device = "cuda"
print(f"Using device: {device}")

# Model names for 4 different BERT variants
model_name = [
    "bert-base-uncased",                # BERT base
    "microsoft/deberta-v3-large",       # DeBERTa v3 Large
    "answerdotai/ModernBERT-base",      # ModernBERT base
    "chandar-lab/NeoBERT"               # NeoBERT
]

# Load all 4 models and tokenizers
# print("Loading BERT base...")
# bert_tokenizer = AutoTokenizer.from_pretrained(model_name[0])
# bert_model = AutoModelForSequenceClassification.from_pretrained(
#     model_name[0],
#     num_labels=3
# ).to(device)

#bert_model = get_peft_model(bert_model, lora_config)

print("Loading DeBERTa v3 Large...")
deberta_tokenizer = AutoTokenizer.from_pretrained(model_name[1])
deberta_model = AutoModelForSequenceClassification.from_pretrained(
    model_name[1],
    num_labels=3
).to(device)

# print("Loading ModernBERT base...")
# modernbert_tokenizer = AutoTokenizer.from_pretrained(model_name[2])
# modernbert_model = AutoModelForSequenceClassification.from_pretrained(
#     model_name[2],
#     num_labels=3
# ).to(device)

# print("Loading NeoBERT...")
# neobert_tokenizer = AutoTokenizer.from_pretrained(model_name[3])
# neobert_model = AutoModelForSequenceClassification.from_pretrained(
#     model_name[3],
#     num_labels=3
# ).to(device)

print("All models loaded successfully!")

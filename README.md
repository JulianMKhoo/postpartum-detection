# Postpartum Depression Detection from Social Media Text

A machine learning system for screening postpartum depression from Reddit posts using transformer-based models.

⚠️ **Disclaimer**: This is a research/educational project using public social media data. It is NOT a clinical diagnostic tool and should not replace professional medical assessment.

---

## 🎯 Overview

Postpartum depression (PPD) affects 10-15% of new mothers but often goes undetected. This project explores whether natural language processing can identify potential PPD cases from social media text, enabling early intervention.

**Key Results:**

- **96.16% accuracy** on 3-class classification
- **95-96% precision/recall** across all classes
- Confidence-based flagging system (5% of cases flagged for review)
- Tested on 18,000+ Reddit posts

---

## 📊 Problem Statement

**Research Question:** Can we detect postpartum depression from informal social media text?

**Challenges:**

- Noisy, informal language (sarcasm, slang, typos)
- No clinical validation of labels
- Class overlap (general depression vs. postpartum-specific)
- Privacy and ethical considerations

---

## 🧪 Methodology

### Dataset

**Source:** Reddit posts from mental health and pregnancy subreddits

- **Normal Pregnancy:** r/BabyBumps, r/pregnant (N=82,503)
- **General Depression:** r/depression, r/SuicideWatch (N=82,503)
- **Postpartum Depression:** r/PostpartumDepression (N=82,503)

**Preprocessing:**

- Removed `[deleted]` and `[removed]` posts
- Filtered posts shorter than 10 characters
- **Cross-contamination removal:**
  - Excluded pregnancy-related keywords from general depression subreddits
  - Excluded depression-related keywords from pregnancy subreddits
- Balanced to 82,503 samples per class (247,509 total)

**Split:**

- Train: 80%
- Validation: 10%
- Test: 10%

### Model

**Architecture:** DeBERTa-v3-large

- Pre-trained transformer with 304M parameters
- Fine-tuned for 3-class classification
- Max sequence length: 512 tokens

**Training:**

- Optimizer: AdamW (lr=5e-5, weight_decay=0.01)
- Batch size: 256
- Epochs: 3
- Hardware: NVIDIA GPU (CUDA)
- Early stopping based on validation F1-macro

**Evaluation Metrics:**

- Accuracy, Precision, Recall, F1-score (macro & weighted)
- Confusion matrix
- Confidence gap analysis (for clinical safety)

---

## 📈 Results

### Overall Performance

| Metric                | Score  |
| --------------------- | ------ |
| **Accuracy**          | 96.16% |
| **F1-Score (Macro)**  | 96.16% |
| **Precision (Macro)** | 96.17% |
| **Recall (Macro)**    | 96.16% |

### Per-Class Performance

| Class                 | Precision | Recall | F1-Score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Normal Pregnancy      | 95.55%    | 95.87% | 95.71%   | 6,143   |
| General Depression    | 96.62%    | 95.87% | 96.24%   | 6,144   |
| Postpartum Depression | 96.32%    | 96.76% | 96.54%   | 6,144   |

### Confidence Analysis

**High Confidence Predictions:** 95.20% of cases

- Mean confidence gap: 95.33%
- Median confidence gap: 99.93%

**Uncertain Predictions (Flagged for Review):** 4.80% of cases

- Among flagged cases: 60.23% were correct
- These cases should be referred to healthcare professionals

![Confusion Matrix](results/confusion_matrix.png)

---

## 🚨 Limitations

### 1. Data Source

- **Not clinically validated:** Labels based on subreddit membership, not professional diagnosis
- **Self-reported:** Users may exaggerate, seek attention, or misrepresent symptoms
- **Sampling bias:** Reddit users ≠ general population (younger, more tech-savvy, Western)

### 2. Generalization

- **Trained on informal text:** Performance may degrade on clinical documentation
- **English only:** Does not support other languages
- **Cultural bias:** Training data predominantly from Western, English-speaking users

### 3. Clinical Applicability

- **Screening tool ONLY:** Cannot replace clinical assessment (PHQ-9, EPDS)
- **No temporal information:** Cannot track symptom progression over time
- **Missing context:** No access to medical history, medications, social support

### 4. Expected Performance Drop

If deployed on clinical populations:

- **Best case:** 85-90% accuracy (similar language patterns)
- **Worst case:** 70-80% accuracy (very different text style)
- **Would require domain adaptation and clinical validation**

---

## 💡 Clinical Context

### Comparison to Standard Screening Tools

| Tool           | Sensitivity | Specificity | Context                 |
| -------------- | ----------- | ----------- | ----------------------- |
| PHQ-9          | 80-85%      | 90-92%      | Clinical setting        |
| EPDS           | 75-85%      | 78-88%      | Postpartum screening    |
| **This Model** | **96%**     | **96%**     | Social media (informal) |

**Note:** Direct comparison is misleading due to different data sources and use cases.

### Use Case

This model is designed for:

- ✅ Social media monitoring for at-risk individuals
- ✅ Early warning system (flag high-risk posts)
- ✅ Research on language patterns in mental health

**NOT for:**

- ❌ Clinical diagnosis
- ❌ Treatment decisions
- ❌ Legal/insurance purposes

---

## 🚀 Usage

### Installation

```bash
Clone repository
git clone https://github.com/yourusername/postpartum-detection
cd postpartum-detectionInstall dependencies
pip install -r requirements.txt
```

### Training

```bash
Prepare data
python src/tools.py
python src/clean_dataset.py
python src/dataset.pyTrain model
python src/main.py --epochs 3 --batch 256
```

### Evaluation

```bashGenerate results and confusion matrix
python src/result.py
```

### Demo (Streamlit)

```bash
streamlit run app/app.py
## 📁 Project Structurepostpartum-depression-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── clean_dataset.py    # Data preprocessing
│   ├── dataset.py           # Dataset class
│   ├── main.py              # Training script
│   ├── model.py             # Model definitions
│   ├── result.py            # Evaluation & analysis
│   ├── tools.py             # Data extraction
│   └── utils.py             # Helper functions
├── app/
│   └── app.py               # Streamlit deployment
├── results/
│   └── confusion_matrix.png
└── best_model/              # Saved model weights
```

## 🔮 Future Work

1. **Multi-language support:** Extend to Spanish, French, Thai
2. **Temporal modeling:** Track symptom progression over time
3. **Multi-modal:** Incorporate post metadata (time, engagement)
4. **Clinical validation:** Partner with healthcare providers
5. **Explainability:** Add LIME/SHAP for interpretability
6. **Real-time monitoring:** Deploy as social media alert system

## 📚 References

1. Despotis, A., et al. (2025). _Suicidal ideation in the postpartum period._ Journal of Affective Disorders. [DOI](https://doi.org/10.1016/j.jad.2025.119707)

2. Devlin, J., et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding._ arXiv. [Link](https://arxiv.org/abs/1810.04805)

3. Sun, C., et al. (2020). _How to Fine-Tune BERT for Text Classification?_ arXiv. [Link](https://arxiv.org/abs/1905.05583)

## ⚖️ Ethics & Privacy

- All data from public Reddit posts (no private messages)
- No personally identifiable information (PII) collected
- Subreddit names anonymized in deployment
- Model outputs include uncertainty quantification
- Clear disclaimers about non-diagnostic nature

## 📝 License

MIT License - See LICENSE file

## 👤 Author

Julian Khoo [GitHub](https://github.com/JulianMKhoo) | [LinkedIn](https://linkedin.com/in/jaratphongmee)

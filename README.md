---

## Data Processing Pipeline

1. **Schema normalization**  
2. **Stratified sampling** for balanced labels  
3. **Text preprocessing using spaCy**
   - tokenization
   - lemmatization
   - stopword removal
4. Output stored as clean, model-ready CSVs

Pipeline is fully reproducible via scripts in `src/`.

---

## Baseline Model

**TF-IDF + Logistic Regression**

- Unigrams + bigrams
- 100k max features
- L2-regularized Logistic Regression

This model serves as a strong classical NLP baseline.

---

## Results

### Overall Performance
- Accuracy: **81.4%**
- Macro F1: **0.81**

### Per-Source Accuracy

| Source | Accuracy |
|------|----------|
| Reviews | **87.8%** |
| Social Media | **75.2%** |
| Support Tickets | **49.2%** |

---

## Error Analysis & Insights

Feature inspection shows expected sentiment cues:

**Positive**
- excellent, great, highly recommend, perfect

**Negative**
- disappointing, poor, terrible, waste, awful

However, error analysis reveals:

- Support tickets often contain **procedural and emotionally neutral language**
- Sentiment is frequently **implicit**, not lexical
- Classical bag-of-words models struggle under domain shift

This motivates the use of **contextual transformer models**.

---

## Key Takeaways

- Classical NLP models generalize poorly across domains
- Feature inspection is critical for diagnosing failures
- Support tickets require contextual understanding beyond keywords

---

## Next Steps

- Fine-tune DistilBERT for multi-source sentiment classification
- Compare improvements, especially on support tickets
- Add domain-aware evaluation

---

## Tech Stack

- Python
- pandas, NumPy
- spaCy
- scikit-learn
- TF-IDF, Logistic Regression

---

## Author

Somana  
MS in Computer Science  

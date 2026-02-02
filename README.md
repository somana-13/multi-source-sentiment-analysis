# Multi-Source Sentiment Analysis (NLP)

An end-to-end NLP project that builds and evaluates sentiment classification models across **multiple real-world text sources**:
- product reviews
- social media posts
- customer support tickets

The project focuses on **domain shift** — how models trained on one type of text degrade when applied to others — and uses systematic error analysis to motivate more advanced models.

---

## Problem Statement

Sentiment analysis models often perform well on a single dataset (e.g., Amazon reviews) but fail when deployed on different text distributions such as support tickets or social media.

This project investigates:
- how classical NLP models behave across domains
- where they fail
- why contextual models are needed for implicit sentiment

---

## Data Overview

Three heterogeneous text sources were used and normalized to a common schema:

`text | label | source`

| Source | Description | Sample Size |
|------|------------|-------------|
| Reviews | Product reviews | 200,000 |
| Social Media | Tweets (Sentiment140) | 200,000 |
| Support Tickets | Customer issue descriptions | 2,189 |

> Raw and processed datasets are excluded from the repo for size reasons.

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
- Support tickets often contain **procedural, emotionally neutral language**
- Sentiment is frequently **implicit**, not lexical
- Classical bag-of-words models struggle under domain shift

This motivates the use of **contextual transformer models**.

---

## How to Run

```bash
# install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# normalize datasets
python src/load_reviews.py
python src/load_social.py
python src/load_tickets.py

# sample + preprocess
python src/sample_data.py
python src/run_preprocessing.py

# build training data and train baseline
python src/build_train_data.py
python src/train_baseline.py

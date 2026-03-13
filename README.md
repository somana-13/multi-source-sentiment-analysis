# Multi-Source Sentiment Analysis (NLP)

This project compares a strong classical NLP baseline with a transformer-based model to evaluate how different approaches handle sentiment across multiple domains.

Baseline: TF-IDF + Logistic Regression

A classical bag-of-words pipeline was implemented using:
	•	TF-IDF vectorization
	•	unigrams + bigrams
	•	100k maximum features
	•	L2-regularized Logistic Regression

This approach serves as a strong baseline for sentiment classification tasks and performs particularly well when sentiment is expressed through explicit polarity words.

⸻

Transformer Model: DistilBERT

To evaluate contextual language models, DistilBERT was fine-tuned for binary sentiment classification.

Configuration
	•	Model: distilbert-base-uncased
	•	Training samples: 60,000
	•	Epochs: 3
	•	Batch size: 8
	•	Dynamic padding via DataCollatorWithPadding

Unlike bag-of-words models, transformers use contextual embeddings that capture relationships between words, which can help when sentiment is expressed implicitly.
Despite using significantly fewer training samples, DistilBERT achieved comparable performance to the classical baseline.
Key Insights

Classical models remain strong on explicit sentiment

Product reviews often contain clear sentiment cues such as:
	•	“excellent”
	•	“terrible”
	•	“highly recommend”

Bag-of-words models capture these patterns effectively, explaining the strong performance of TF-IDF on review data.

⸻

Transformers improve implicit sentiment understanding

Support tickets frequently express dissatisfaction indirectly, for example:

“The application crashes whenever I try to export the report.”

This type of procedural language contains implicit negative sentiment.

DistilBERT improved performance on this domain:

Support ticket accuracy
TF-IDF baseline: 49.2%
DistilBERT: 54.1%

This suggests contextual models better capture sentiment when it is not expressed with obvious polarity words.

Takeaways
	•	Classical models remain highly competitive when sentiment is expressed through explicit lexical cues.
	•	Transformer models provide advantages when sentiment must be inferred from context or procedural descriptions.
	•	Domain distribution and text style play a significant role in model performance.

⸻

Project Goal

The goal of this project is to evaluate how different NLP modeling approaches perform across heterogeneous text sources, including:
	•	structured product reviews
	•	informal social media posts
	•	procedural customer support tickets

This setup highlights how model performance varies across domains and where contextual language models provide the greatest benefit.
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
DistilBERT (sample=60k, epochs=3)

Accuracy: 0.8097
F1: 0.8136

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

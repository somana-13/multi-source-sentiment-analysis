# Multi-Source Sentiment Analysis (NLP)

End-to-end NLP project that builds and evaluates sentiment classification models across **multiple real-world text sources**: customer reviews, social media posts, and customer support tickets.

The project emphasizes:
- clean data pipelines
- strong baselines
- error analysis
- domain generalization challenges in NLP

---

## Problem Statement

Sentiment analysis models often perform well on a single domain (e.g., product reviews) but degrade when applied to other text sources such as support tickets or social media.

This project investigates:
- how classical NLP models behave across domains
- where they fail
- why contextual models are needed for implicit sentiment

---

## Data Sources

Three heterogeneous text sources were used:

| Source | Description | Size (sampled) |
|------|------------|----------------|
| Reviews | Amazon product reviews | 200,000 |
| Social Media | Tweets (Sentiment140) | 200,000 |
| Support Tickets | Customer issue descriptions | 2,189 |

Each dataset was normalized to a common schema:

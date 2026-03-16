######Multi-Source Sentiment Analysis##########

This project builds an end-to-end NLP system that analyzes sentiment across multiple text domains: product reviews, social media posts, and customer support tickets.

Two modeling approaches are implemented and compared:
	•	a classical NLP baseline using TF-IDF and Logistic Regression
	•	a transformer model using DistilBERT

The project also includes an interactive Streamlit web application that allows users to enter text and see real-time sentiment predictions.
--------------------------------------
Project Motivation

Sentiment analysis models are often trained on a single type of text, but real-world applications must handle multiple domains.

Language varies significantly depending on the source:
	•	Product reviews often contain explicit sentiment words such as “excellent” or “terrible”.
	•	Social media posts tend to be informal and short.
	•	Support tickets typically describe problems rather than express opinions directly.

This project explores how different modeling approaches handle these differences and how well they generalize across domains.
------------------------------------
Data Sources

Three datasets were used.

Amazon product reviews provide large amounts of labeled sentiment text with explicit opinions.

Sentiment140 tweets provide social media data with informal language and short messages.

Customer support tickets simulate technical issue descriptions where sentiment is often implicit.

All datasets were normalized into a unified schema with the following fields:

text – the input text
label – sentiment (0 for negative, 1 for positive)
source – the dataset origin (reviews, social, or tickets)

⸻

Data Processing Pipeline

The data pipeline performs several preprocessing steps:
	•	schema normalization across datasets
	•	text preprocessing using spaCy
	•	tokenization and lemmatization
	•	stopword removal
	•	creation of a unified training dataset

The processed data is then used for both classical and transformer-based models.
------------------------------------
Models

Baseline Model

The classical baseline uses TF-IDF vectorization combined with Logistic Regression.
The vectorizer includes unigrams and bigrams and limits the vocabulary to the most informative features.

This approach performs well when sentiment is expressed using explicit polarity words.

Transformer Model

A DistilBERT transformer model was fine-tuned for binary sentiment classification.

Configuration:

model: distilbert-base-uncased
training samples: 60,000
epochs: 3
batch size: 8

Transformers capture contextual relationships between words and are expected to perform better when sentiment must be inferred from context.

⸻

Results and Observations

Both models achieve similar overall performance.

However, important differences appear across text domains.

The TF-IDF baseline performs very well on product reviews because reviews contain strong sentiment vocabulary.

DistilBERT performs slightly better on social media posts where language is less structured.

The largest improvement appears on support ticket data.
Support tickets usually describe technical issues rather than directly expressing sentiment.

For example:

“The application crashes whenever I try to export the report.”

This sentence does not contain explicit negative words, but the underlying sentiment is clearly negative.

The transformer model is better at detecting this type of implicit sentiment.
----
Streamlit Demo

The repository includes an interactive Streamlit web application for live sentiment prediction.

Run the demo locally with:

python3 -m streamlit run app.py

Users can enter text, select the source type, and view the predicted sentiment along with model confidence.
------------------------------------
Project Structure

multi_source_sentiment

data
raw datasets and processed datasets

src
data loading scripts
preprocessing pipeline
baseline training
transformer training

assets
demo screenshots

app.py
Streamlit application

README.md
project documentation

requirements.txt
project dependencies
----------------------------------------
Reproducing the Experiments

Install dependencies:

pip install -r requirements.txt

Prepare datasets:

python src/load_reviews.py
python src/load_social.py
python src/load_tickets.py
python src/run_preprocessing.py

Train the baseline model:

python src/train_baseline.py

Train the transformer model:

python src/sample_for_bert.py
python src/train_distilbert.py
---------------------------------------------
Technologies Used

Python
Pandas
scikit-learn
spaCy
Hugging Face Transformers
PyTorch
Streamlit
import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

OUTPUT_DIR = "distilbert-sentiment-output"

def get_latest_checkpoint(output_dir):
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        raise ValueError("No checkpoint folders found.")
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

MODEL_PATH = get_latest_checkpoint(OUTPUT_DIR)
#MODEL_PATH = "distilbert-sentiment-output/checkpoint-18000"

st.set_page_config(page_title="Multi-Source Sentiment Analysis", page_icon="🧠")

st.title("🧠 Multi-Source Sentiment Analysis")
st.caption("DistilBERT fine-tuned on reviews, social media, and support tickets")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()
    label_map = {0: "Negative", 1: "Positive"}
    return label_map[pred], confidence

tokenizer, model = load_model()

with st.sidebar:
    st.header("Try examples")
    st.write("**Reviews**")
    st.write("- This product is amazing and works perfectly.")
    st.write("- Terrible quality, complete waste of money.")
    st.write("**Tickets**")
    st.write("- The application crashes every time I export the report.")
    st.write("- My issue was resolved quickly and everything works now.")
    st.write("**Social**")
    st.write("- absolutely love this")
    st.write("- this is so frustrating")

source = st.selectbox(
    "Choose text source",
    ["reviews", "social", "tickets"]
)

text = st.text_area("Enter text", height=150)

if st.button("Predict Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        prediction, confidence = predict_sentiment(text, tokenizer, model)

        st.subheader("Prediction")
        st.write(f"**Source:** {source}")
        st.write(f"**Sentiment:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2%}")

        if prediction == "Positive":
            st.success("The model predicts a positive sentiment.")
        else:
            st.error("The model predicts a negative sentiment.")
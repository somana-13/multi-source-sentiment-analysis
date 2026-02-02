import sys
from pathlib import Path

# Add src/ to Python path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from preprocess import NLP, normalize_basic, clean_doc

FILES_IN = [
    "data/processed/reviews_sample.csv",
    "data/processed/tickets_sample.csv",
    "data/processed/social_sample.csv",
]

def preprocess_file(path_in: str):
    df = pd.read_csv(path_in)

    # Basic normalization first (cheap)
    texts = df["text"].astype(str).map(normalize_basic).tolist()

    # spaCy pipe (much faster than row-by-row)
    cleaned = []
    for doc in NLP.pipe(texts, batch_size=2000, n_process=4):
        cleaned.append(clean_doc(doc))

    df["clean_text"] = cleaned
    empty_rate = (df["clean_text"].str.len() == 0).mean()
    
    print(f"Empty clean_text rate: {empty_rate:.3%}")
    
    path_out = path_in.replace("_sample.csv", "_sample_clean.csv")
    df.to_csv(path_out, index=False)

    print("Saved:", path_out, "| rows:", len(df))

def main():
    for f in FILES_IN:
        preprocess_file(f)

if __name__ == "__main__":
    main()
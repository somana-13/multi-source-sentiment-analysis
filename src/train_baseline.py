import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/processed/train_data_clean.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df["clean_text"].astype(str)
    y = df["label"].astype(int)

    # Keep indices so we can do per-source evaluation later
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_vec, y_train)

    import numpy as np

    feature_names = tfidf.get_feature_names_out()
    weights = clf.coef_[0]

    top_k = 20

    top_pos_idx = np.argsort(weights)[-top_k:]
    top_neg_idx = np.argsort(weights)[:top_k]

    print("\n=== Top Positive Features ===")
    for i in reversed(top_pos_idx):
       print(f"{feature_names[i]:25s} {weights[i]:.3f}")

    print("\n=== Top Negative Features ===")
    for i in top_neg_idx:
       print(f"{feature_names[i]:25s} {weights[i]:.3f}")

    preds = clf.predict(X_val_vec)

    print("\n=== Overall Validation Report ===")
    print(classification_report(y_val, preds, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_val, preds))

    # Per-source evaluation
    val_df = df.loc[X_val.index, ["source", "label"]].copy()
    val_df["pred"] = preds

    # Attach predictions to validation data
    val_full = df.loc[X_val.index].copy()
    val_full["pred"] = preds

   # Focus specifically on tickets (your weak spot)
    tickets_val = val_full[val_full["source"] == "tickets"]

    fp_tickets = tickets_val[(tickets_val["label"] == 0) & (tickets_val["pred"] == 1)]
    fn_tickets = tickets_val[(tickets_val["label"] == 1) & (tickets_val["pred"] == 0)]

    print("\n=== Tickets: False Positives (Neg → Pos) ===")
    for _, r in fp_tickets.sample(min(5, len(fp_tickets)), random_state=42).iterrows():
      print(r["clean_text"][:200])

    print("\n=== Tickets: False Negatives (Pos → Neg) ===")
    for _, r in fn_tickets.sample(min(5, len(fn_tickets)), random_state=42).iterrows():
      print(r["clean_text"][:200])

    print("\n=== Per-Source Accuracy ===")
    for src in sorted(val_df["source"].unique()):
        sub = val_df[val_df["source"] == src]
        acc = (sub["pred"] == sub["label"]).mean()
        print(f"{src:8s} | n={len(sub):6d} | acc={acc:.4f}")

if __name__ == "__main__":
    main()
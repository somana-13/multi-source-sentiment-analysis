import pandas as pd

FILES = [
    "data/processed/reviews_sample_clean.csv",
    "data/processed/tickets_sample_clean.csv",
    "data/processed/social_sample_clean.csv",
]

OUT_PATH = "data/processed/train_data_clean.csv"

def main():
    dfs = [pd.read_csv(f) for f in FILES]
    df = pd.concat(dfs, ignore_index=True)

    # Expect these columns to exist
    df = df[["clean_text", "label", "source"]].dropna()
    df = df[df["clean_text"].astype(str).str.len() > 0].copy()

    df["label"] = df["label"].astype(int)

    df.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH, "| shape:", df.shape)
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("\nSource distribution:\n", df["source"].value_counts())

if __name__ == "__main__":
    main()
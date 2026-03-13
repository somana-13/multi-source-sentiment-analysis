import pandas as pd

INPUT_PATH = "data/processed/train_data_clean.csv"
OUTPUT_PATH = "data/processed/train_data_bert_sample.csv"

N_PER_CLASS = 30000
SEED = 42

def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep only the columns we need
    df = df[["clean_text", "label", "source"]].dropna().copy()
    df["label"] = df["label"].astype(int)

    sampled_parts = []
    for label_value, group in df.groupby("label"):
        n_take = min(N_PER_CLASS, len(group))
        sampled_parts.append(group.sample(n=n_take, random_state=SEED))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)

    sampled.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("\nLabel distribution:")
    print(sampled["label"].value_counts())
    print("\nSource distribution:")
    print(sampled["source"].value_counts())
    print("\nColumns:")
    print(sampled.columns.tolist())

if __name__ == "__main__":
    main()
import pandas as pd

INPUT_PATH = "data/raw/social/training.1600000.processed.noemoticon.csv"
OUTPUT_PATH = "data/processed/social.csv"

def main():
    df = pd.read_csv(INPUT_PATH, header=None, encoding="latin-1")

    # Keep only sentiment + text
    df = df[[0, 5]]
    df.columns = ["label", "text"]

    # Normalize labels: 0 -> 0 (neg), 4 -> 1 (pos)
    df["label"] = df["label"].map({0: 0, 4: 1})

    # Drop any unexpected rows
    df = df.dropna()

    df["source"] = "social"

    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Label distribution:")
    print(df["label"].value_counts())
    print("\nExample tweet:")
    print(df["text"].iloc[0])

if __name__ == "__main__":
    main()
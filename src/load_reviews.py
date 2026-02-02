import pandas as pd

TRAIN_PATH = "data/raw/amazon_reviews/train.csv"
TEST_PATH = "data/raw/amazon_reviews/test.csv"
OUTPUT_PATH = "data/processed/reviews.csv"

def load_and_clean(path):
    # No header in file
    df = pd.read_csv(path, header=None)
    df.columns = ["label", "title", "text"]

    # Combine title + review body
    df["text"] = df["title"] + ". " + df["text"]

    # Normalize labels: 1 -> 0 (negative), 2 -> 1 (positive)
    df["label"] = df["label"].map({1: 0, 2: 1})

    # Add source column
    df["source"] = "reviews"

    return df[["text", "label", "source"]]

def main():
    train_df = load_and_clean(TRAIN_PATH)
    test_df = load_and_clean(TEST_PATH)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Label distribution:")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
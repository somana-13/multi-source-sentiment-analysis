import pandas as pd

TRAIN_PATH = "data/raw/amazon_reviews/train.csv"
TEST_PATH = "data/raw/amazon_reviews/test.csv"

def inspect(path, name):
    print(f"\n--- {name} ---")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Label distribution (first 10k rows):")
    print(df.iloc[:10000, 0].value_counts())
    print("\nSample rows:")
    print(df.head(3))

if __name__ == "__main__":
    inspect(TRAIN_PATH, "TRAIN")
    inspect(TEST_PATH, "TEST")

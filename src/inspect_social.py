import pandas as pd

SOCIAL_PATH = "data/raw/social/training.1600000.processed.noemoticon.csv"

def main():
    df = pd.read_csv(SOCIAL_PATH, header=None, encoding="latin-1")

    print("Shape:", df.shape)
    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nLabel distribution (first 10k rows):")
    print(df.iloc[:10000, 0].value_counts())

if __name__ == "__main__":
    main()
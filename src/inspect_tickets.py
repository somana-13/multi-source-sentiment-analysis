import pandas as pd

TICKETS_PATH = "data/raw/support_tickets/tickets.csv"  # <-- change to your real filename

def main():
    df = pd.read_csv(TICKETS_PATH)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\nSample rows:")
    print(df.head(3))

if __name__ == "__main__":
    main()
import pandas as pd

INPUT_PATH = "data/raw/support_tickets/tickets.csv"   # <-- update if your filename differs
OUTPUT_PATH = "data/processed/tickets.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # 1) Build text field (subject + description)
    df["text"] = df["Ticket Subject"].astype(str) + ". " + df["Ticket Description"].astype(str)

    # 2) Keep only rows with satisfaction rating
    df = df[df["Customer Satisfaction Rating"].notna()].copy()

    # 3) Convert rating to numeric (in case it's float like 3.0)
    df["Customer Satisfaction Rating"] = pd.to_numeric(df["Customer Satisfaction Rating"], errors="coerce")
    df = df[df["Customer Satisfaction Rating"].notna()].copy()

    # 4) Drop neutral rating 3
    df = df[df["Customer Satisfaction Rating"] != 3].copy()

    # 5) Normalize to binary label: >=4 positive(1), <=2 negative(0)
    df["label"] = (df["Customer Satisfaction Rating"] >= 4).astype(int)

    # 6) Add source + keep final schema
    df["source"] = "tickets"
    out = df[["text", "label", "source"]].dropna()

    out.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)
    print("Rows:", len(out))
    print("Label distribution:")
    print(out["label"].value_counts())
    print("\nExample text:")
    print(out["text"].iloc[0][:200], "...")
    
if __name__ == "__main__":
    main()
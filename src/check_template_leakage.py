import pandas as pd

df = pd.read_csv("data/processed/train_data_clean.csv")

suspects = ["note", "seller", "responsible", "disclaimer", "complaint", "will disappoint"]

for term in suspects:
    mask = df["clean_text"].str.contains(term, na=False)
    if mask.sum() == 0:
        continue
    rate_pos = df.loc[mask, "label"].mean()
    print(f"{term:15s} | count={mask.sum():6d} | positive_rate={rate_pos:.3f}")
import pandas as pd

REVIEWS_IN = "data/processed/reviews.csv"
SOCIAL_IN  = "data/processed/social.csv"
TICKETS_IN = "data/processed/tickets.csv"

REVIEWS_OUT = "data/processed/reviews_sample.csv"
SOCIAL_OUT  = "data/processed/social_sample.csv"
TICKETS_OUT = "data/processed/tickets_sample.csv"

SEED = 42
REVIEWS_N = 200_000
SOCIAL_N  = 200_000

REQUIRED_COLS = ["text", "label", "source"]

def ensure_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    # Social may be ['label','text','source'] — reorder consistently
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}. Found: {df.columns.tolist()}")
    return df[REQUIRED_COLS].copy()

def stratified_sample(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """
    Stratified sample that always preserves columns (including 'label').
    Avoids groupby.apply() quirks by sampling indices per class.
    """
    if n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    per_class = n // 2

    idx_parts = []
    for lab, g in df.groupby("label"):
        take = min(per_class, len(g))
        idx_parts.append(g.sample(n=take, random_state=seed).index)

    idx = idx_parts[0].append(idx_parts[1]) if len(idx_parts) == 2 else idx_parts[0]

    sampled = df.loc[idx].copy()

    # If we didn't reach n (rare), top up from remaining rows
    if len(sampled) < n:
        remaining = df.drop(sampled.index)
        extra = remaining.sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra], ignore_index=False)

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
def main():
    
    reviews = ensure_schema(pd.read_csv(REVIEWS_IN), "reviews.csv")
    social  = ensure_schema(pd.read_csv(SOCIAL_IN),  "social.csv")
    tickets = ensure_schema(pd.read_csv(TICKETS_IN), "tickets.csv")

    reviews_sample = stratified_sample(reviews, REVIEWS_N)
    social_sample  = stratified_sample(social, SOCIAL_N)
    tickets_sample = tickets  # keep all

    reviews_sample.to_csv(REVIEWS_OUT, index=False)
    social_sample.to_csv(SOCIAL_OUT, index=False)
    tickets_sample.to_csv(TICKETS_OUT, index=False)

    print("Saved samples:")
    print("reviews :", reviews_sample.shape, reviews_sample.columns.tolist())
    print("social  :", social_sample.shape,  social_sample.columns.tolist())
    print("tickets :", tickets_sample.shape, tickets_sample.columns.tolist())

if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd

# rating: 1 again, 2 hard, 3 good, 4 easy
# state: 0 new, 1 learning, 2 review, 3 relearning

FEATURES = [
    "reviews_per_day",
    "consistency",
    "new_cards_per_day",
    "lapse_rate",
    "learning_again_rate",
    "hard_rate",
    "easy_rate",
    "median_seconds",
]


def load_user(path):
    df = pd.read_parquet(path)
    df["user_id"] = int(Path(path).parent.name.split("=")[1])
    return df


def user_features(df):
    df = df[df["rating"].between(1, 4)]

    active_days = df["day_offset"].nunique()
    span_days = df["day_offset"].max() - df["day_offset"].min() + 1

    review = df[df["state"] == 2]
    learning = df[df["state"].isin([0, 1])]
    passed = df[df["rating"] > 1]

    return {
        "user_id": df["user_id"].iloc[0],
        "total_reviews": len(df),
        "active_days": active_days,
        "reviews_per_day": len(df) / active_days,
        "consistency": active_days / span_days,
        "new_cards_per_day": (df["state"] == 0).sum() / active_days,
        "lapse_rate": (review["rating"] == 1).mean() if len(review) else np.nan,
        "learning_again_rate": (learning["rating"] == 1).mean() if len(learning) else np.nan,
        "hard_rate": (passed["rating"] == 2).mean(),
        "easy_rate": (passed["rating"] == 4).mean(),
        "median_seconds": df["duration"].median() / 1000,
    }


def build_feature_table(data_dir="data/revlogs", min_active_days=30):
    rows = []
    paths = sorted(Path(data_dir).glob("user_id=*/data.parquet"))
    for i, p in enumerate(paths, start=1):
        df = load_user(p)
        rows.append(user_features(df))
        if i % 50 == 0:
            print(f"processed {i}/{len(paths)} users")

    table = pd.DataFrame(rows)
    before = len(table)
    table = table[table["active_days"] >= min_active_days].dropna(subset=FEATURES)
    print(f"kept {len(table)} of {before} users (need {min_active_days}+ active days)")
    return table.reset_index(drop=True)

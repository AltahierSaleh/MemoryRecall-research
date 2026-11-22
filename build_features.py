from pathlib import Path

from src.features import build_feature_table

table = build_feature_table("data/revlogs")
Path("output").mkdir(exist_ok=True)
table.to_csv("output/features.csv", index=False)
print("saved output/features.csv")
print(table.describe().round(3).T)

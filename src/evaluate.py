"""Summarise the actual classifier-vs-HMM experiment CSV."""
from pathlib import Path
import pandas as pd

INPUT = Path(__file__).resolve().parents[1] / "results" / "hmm_comparison.csv"

def main():
    df = pd.read_csv(INPUT)
    columns = [
        "classifier_top1_accuracy", "top1_accuracy",
        "classifier_route_continuity", "route_continuity"
    ]
    for col in columns:
        if col in df:
            print(f"{col}: {df[col].mean():.6f}")
    print(f"Trips evaluated: {len(df)}")

if __name__ == "__main__":
    main()

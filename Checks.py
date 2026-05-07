import pandas as pd
from pathlib import Path

UDB_FILES = {
    2017: Path("output/ES_2017_a2.txt"),
    2018: Path("output/ES_2018_a1.txt"),
    2019: Path("output/ES_2019_b1.txt"),
}

def load_udb(path):
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df

deh_labels = {
    0: "no education",
    1: "primary",
    2: "lower secondary",
    3: "upper secondary",
    4: "post-secondary non-tertiary",
    5: "tertiary",
}

for year, path in UDB_FILES.items():
    df = load_udb(path)
    print(f"\nYear {year} — deh distribution:")
    counts = df["deh"].value_counts().sort_index()
    for val, count in counts.items():
        label = deh_labels.get(int(val), "unknown")
        pct = 100 * count / len(df)
        print(f"  deh={val:.0f} ({label:<28}): {count:>6} ({pct:.1f}%)")
    print(f"  Unique values: {sorted(df['deh'].dropna().unique().tolist())}")
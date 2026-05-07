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

# Also check the raw Tp file directly
print("=== RAW TP FILE — PE040 values ===")
for year in [2017, 2018, 2019]:
    try:
        tp_path = Path(f"input_data/ECV_Tp_{year}.dta")
        if tp_path.exists():
            import pandas as pd
            tp = pd.read_stata(str(tp_path), convert_categoricals=False)
            tp.columns = [c.upper() for c in tp.columns]
            if "PE040" in tp.columns:
                print(f"\nYear {year} — PE040 value counts (raw Tp file):")
                print(tp["PE040"].value_counts().sort_index().to_string())
                print(f"Unique values: {sorted(tp['PE040'].dropna().unique().tolist())}")
            else:
                print(f"Year {year}: PE040 not found in Tp file")
    except Exception as e:
        print(f"Year {year}: could not read Tp file — {e}")

print("\n=== UDB OUTPUT — deh values ===")
for year, path in UDB_FILES.items():
    df = load_udb(path)
    print(f"\nYear {year} — deh value counts in UDB:")
    print(df["deh"].value_counts().sort_index().to_string())
    print(f"Unique values: {sorted(df['deh'].dropna().unique().tolist())}")
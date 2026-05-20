import pandas as pd

# Correct file: Td (dwelling/design), not Th (household characteristics)
raw = pd.read_stata(
    "input_data/ECV_Td_2024.dta",
    convert_categoricals=False,
)

# Normalise to uppercase to match how readers.py handles it
raw.columns = [c.upper() for c in raw.columns]

print("Columns in Td file:", [c for c in raw.columns if "DB03" in c or "DB09" in c])

# The analysis dataset stores household_id as string
# DB030 in ECV is numeric — cast to string to match
household_ids = ["1844", "4221"]
raw["DB030_str"] = raw["DB030"].astype(str).str.strip()

match = raw[raw["DB030_str"].isin(household_ids)][["DB030", "DB090"]]
print(match)
print(f"\nTotal rows in Td 2024: {len(raw)}")
print(f"Matching rows found:   {len(match)}")

print(f"DB030 range: {raw['DB030'].min()} to {raw['DB030'].max()}")
print(f"Sample DB030 values:\n{raw['DB030'].head(10).tolist()}")

# Also check: does the raw 2024 Td file have ANY zero-weight rows?
zero_in_raw = raw[raw["DB090"] == 0]
print(f"\nZero-weight rows in raw ECV_Td_2024: {len(zero_in_raw)}")
print(zero_in_raw[["DB030", "DB090"]])
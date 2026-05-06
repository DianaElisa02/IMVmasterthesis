import pandas as pd

def load_euromod_output(path):
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df

df = load_euromod_output("input_data/euromod_output/es_2019_std.txt")

murcia = df[df["drgn2"] == 62].copy()
print(f"Total persons in Murcia (drgn2=62): {len(murcia)}")
print(f"Weighted population: {murcia['dwt'].sum():,.0f}")
print()

# Check bsarg_s distribution
print("bsarg_s distribution in Murcia:")
print(murcia["bsarg_s"].describe())
print(f"Non-zero bsarg_s: {(murcia['bsarg_s'] > 0).sum()}")
print(f"Zero bsarg_s: {(murcia['bsarg_s'] == 0).sum()}")
print(f"NaN bsarg_s: {murcia['bsarg_s'].isna().sum()}")
print()

# Check il_bsarg_62 (Murcia counterfactual)
print("il_bsarg_62 distribution in Murcia:")
print(murcia["il_bsarg_62"].describe())
print(f"Non-zero il_bsarg_62: {(murcia['il_bsarg_62'] > 0).sum()}")
print()

# Check eligibility-related variables for Murcia
print("les distribution in Murcia:")
print(murcia["les"].value_counts().sort_index())
print()
print("dag distribution in Murcia:")
print(murcia["dag"].describe())
print()

# Check income
print("Mean ils_origy in Murcia:", murcia["ils_origy"].mean().round(2))
print("Mean ils_earns in Murcia:", murcia["ils_earns"].mean().round(2))
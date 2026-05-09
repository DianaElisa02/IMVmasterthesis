import pandas as pd
from pathlib import Path
import numpy as np

UDB_FILES = {
    2017: Path("output/ES_2017_a2.txt"),
    2018: Path("output/ES_2018_a1.txt"),
    2019: Path("output/ES_2019_b1.txt"),
}

PROBLEM_REGIONS = {
    23: "La Rioja",
    24: "Aragón",
    63: "Ceuta",
    62: "Murcia",
}

def load_udb(path):
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df

for year, path in UDB_FILES.items():
    df = load_udb(path)
    print(f"\n{'='*70}")
    print(f"YEAR {year}")
    print(f"{'='*70}")

    for drgn2, region in PROBLEM_REGIONS.items():
        reg = df[df["drgn2"] == drgn2].copy()
        wa = reg[(reg["dag"] >= 18) & (reg["dag"] <= 65)].copy()

        print(f"\n--- {region} (drgn2={drgn2}) ---")
        print(f"Total persons: {len(reg):,} | Working age (18-65): {len(wa):,}")
        print(f"Weighted population: {reg['dwt'].sum():,.0f}")

        # 1. IDHH integrity — does household join work?
        print(f"\n[1] Household join integrity:")
        null_hsize = reg["hsize"].isna().sum()
        null_drgn1 = reg["drgn1"].isna().sum()
        null_dwt   = reg["dwt"].isna().sum()
        zero_dwt   = (reg["dwt"] == 0).sum()
        print(f"    hsize nulls: {null_hsize} | drgn1 nulls: {null_drgn1} | dwt nulls: {null_dwt} | dwt zeros: {zero_dwt}")
        if null_hsize > 0 or null_drgn1 > 0:
            print(f"    *** WARNING: null household variables — join may have failed ***")

        # 2. Income test variables — are they plausible?
        print(f"\n[2] Income variables (working age adults):")
        for var in ["yem", "yse", "poa", "bun", "bsa", "yds", "hy020"]:
            if var in wa.columns:
                nonzero = (wa[var] != 0).sum()
                wmean = (wa[var] * wa["dwt"]).sum() / wa["dwt"].sum()
                vmax = wa[var].max()
                nulls = wa[var].isna().sum()
                flag = " ← ALL ZERO" if nonzero == 0 else ""
                flag = " ← SUSPICIOUS" if nonzero < 5 else flag
                print(f"    {var:<8}: non-zero={nonzero:>5}, wmean=€{wmean:>8.2f}, max=€{vmax:>10.2f}, nulls={nulls}{flag}")

        # 3. Employment status — critical for IMV job-seeker condition
        print(f"\n[3] Employment status (les) — working age:")
        les_labels = {0:"child",2:"self-emp",3:"employee",4:"retired",
                      5:"unemployed",6:"student",7:"inactive",8:"disabled",9:"other"}
        for val, count in wa["les"].value_counts().sort_index().items():
            label = les_labels.get(int(val), "?")
            pct = 100 * count / max(len(wa), 1)
            print(f"    les={val:.0f} ({label:<10}): {count:>5} ({pct:.1f}%)")

        # 4. Age distribution — are there enough 23-65 year olds?
        eligible_age = reg[(reg["dag"] >= 23) & (reg["dag"] <= 65)]
        print(f"\n[4] Age eligibility (23-65): {len(eligible_age):,} persons ({100*len(eligible_age)/max(len(reg),1):.1f}%)")
        print(f"    dag mean={reg['dag'].mean():.1f} min={reg['dag'].min():.0f} max={reg['dag'].max():.0f}")

        # 5. Citizenship — dcz must be 1 or 2 for IMV eligibility
        print(f"\n[5] Citizenship (dcz) distribution:")
        for val, count in reg["dcz"].value_counts().sort_index().items():
            pct = 100 * count / max(len(reg), 1)
            label = {1:"national", 2:"EU", 3:"non-EU"}.get(int(val), "?")
            print(f"    dcz={val:.0f} ({label}): {count:>5} ({pct:.1f}%)")

        # 6. Education — deh should now be 0-5
        print(f"\n[6] Education (deh) distribution:")
        for val, count in reg["deh"].value_counts().sort_index().items():
            pct = 100 * count / max(len(reg), 1)
            print(f"    deh={val:.0f}: {count:>5} ({pct:.1f}%)")
        if reg["deh"].max() == 0:
            print(f"    *** WARNING: all deh=0, education recode may have failed ***")

        # 7. Household size — any implausible values?
        print(f"\n[7] Household structure:")
        print(f"    hsize: mean={reg['hsize'].mean():.2f} min={reg['hsize'].min():.0f} max={reg['hsize'].max():.0f}")
        print(f"    Single person HH: {(reg['hsize']==1).sum():,}")
        print(f"    Multi-person HH:  {(reg['hsize']>1).sum():,}")

        # 8. IDHH sample — check a few actual values to verify join worked
        print(f"\n[8] Sample IDHH and idperson values:")
        sample = reg[["idhh","idperson","drgn2","hsize","dwt","dag","les"]].head(5)
        print(sample.to_string(index=False))

        # 9. Zero income working-age adults — potential eligibility pool
        zero_inc = wa[
            (wa["yem"]==0) & (wa["yse"]==0) & (wa["poa"]==0) &
            (wa["bun"]==0) & (wa["bsa"]==0)
        ]
        pct_zero = 100 * len(zero_inc) / max(len(wa), 1)
        print(f"\n[9] Working-age adults with ALL income=0: {len(zero_inc):,} ({pct_zero:.1f}%)")
        print(f"    These are theoretically IMV-eligible on income grounds")
        if pct_zero > 50:
            print(f"    *** HIGH: >50% have zero income — check if income vars populated ***")

        # 10. Compare with national average for same variables
        nat_wa = df[(df["dag"] >= 18) & (df["dag"] <= 65)]
        nat_zero_pct = 100 * len(nat_wa[(nat_wa["yem"]==0)&(nat_wa["yse"]==0)&
                                         (nat_wa["poa"]==0)&(nat_wa["bun"]==0)&
                                         (nat_wa["bsa"]==0)]) / max(len(nat_wa), 1)
        print(f"    National average zero-income working age: {nat_zero_pct:.1f}%")
        diff = pct_zero - nat_zero_pct
        if abs(diff) > 10:
            print(f"    *** DIVERGES from national by {diff:+.1f}pp — investigate ***")
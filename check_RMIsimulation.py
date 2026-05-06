import pandas as pd

INFORME_TITULARES = {
    2017: 313291,
    2018: 293302,
    2019: 297183,
}

FILES = {
    2017: "/workspaces/IMVmasterthesis/input_data/euromod_output/es_2017_std.txt",
    2018: "/workspaces/IMVmasterthesis/input_data/euromod_output/es_2018_std.txt",
    2019: "/workspaces/IMVmasterthesis/input_data/euromod_output/es_2019_std.txt",
}

REGION_NAMES = {
    11: "Galicia", 12: "Asturias", 13: "Cantabria",
    21: "País Vasco", 22: "Navarra", 23: "La Rioja",
    24: "Aragón", 30: "Madrid", 41: "Castilla y León",
    42: "Castilla-La Mancha", 43: "Extremadura",
    51: "Cataluña", 52: "C. Valenciana", 53: "Illes Balears",
    61: "Andalucía", 62: "Murcia", 63: "Ceuta",
    64: "Melilla", 70: "Canarias",
}


def load_euromod_output(path: str) -> pd.DataFrame:
    """
    Reads a EUROMOD output file and robustly converts all columns to numeric,
    handling comma decimals and Arrow-backed string dtypes.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce"
        )
    return df


for year, path in FILES.items():
    print("=" * 60)
    print(f"YEAR {year}")
    print("=" * 60)

    df = load_euromod_output(path)

    # verify key columns loaded correctly
    for key_col in ["dwt", "bsarg_s", "dag", "les", "drgn2"]:
        if df[key_col].dtype not in ["float64", "int64"]:
            print(f"  WARNING: {key_col} dtype is {df[key_col].dtype} — check file")

    # --- 1. Population ---
    total_pop = df["dwt"].sum()
    print(f"\nTotal weighted population: {total_pop:,.0f}")

    # --- 2. Recipient counts ---
    recipients = df[df["bsarg_s"] > 0].copy()
    unweighted = len(recipients)
    weighted = recipients["dwt"].sum()
    informe = INFORME_TITULARES[year]
    ratio = weighted / informe

    print(f"\nRecipient counts:")
    print(f"  Unweighted:              {unweighted:,}")
    print(f"  Weighted (EUROMOD):      {weighted:,.0f}")
    print(f"  Informe RMI titulares:   {informe:,}")
    print(f"  Ratio EUROMOD/Informe:   {ratio:.3f}")

    # --- 3. Regional breakdown ---
    regional = (
        recipients.groupby("drgn2")
        .apply(lambda x: pd.Series({
            "weighted_recipients": round(x["dwt"].sum(), 0),
            "weighted_mean_monthly": round(
                (x["bsarg_s"] * x["dwt"]).sum() / x["dwt"].sum(), 2
            ),
        }))
        .reset_index()
    )
    regional["region"] = regional["drgn2"].map(REGION_NAMES)

    print(f"\nRegional breakdown:")
    print(regional[["region", "drgn2", "weighted_recipients",
                     "weighted_mean_monthly"]].to_string(index=False))

    # --- 4. Implausible value checks ---
    print(f"\nImplausible value checks:")

    neg = (recipients["bsarg_s"] < 0).sum()
    print(f"  Negative bsarg_s:                {neg}")

    underage = recipients[recipients["dag"] < 18]
    print(f"  Recipients dag < 18:             {len(underage)}"
          f" (weighted: {underage['dwt'].sum():,.0f})")

    overage = recipients[recipients["dag"] > 65]
    print(f"  Recipients dag > 65:             {len(overage)}"
          f" (weighted: {overage['dwt'].sum():,.0f})")

    employed = recipients[recipients["les"] == 3]
    print(f"  Recipients les=3 (employed):     {len(employed)}"
          f" (weighted: {employed['dwt'].sum():,.0f})")

    placeholder = regional[regional["weighted_mean_monthly"] <= 1.0]
    if len(placeholder) > 0:
        print(f"  Regions with mean <= €1.00:      "
              f"{placeholder['region'].tolist()}")
    else:
        print(f"  No placeholder regions detected  ✓")

    # --- 5. Aggregate expenditure ---
    total_expenditure = (
        recipients["bsarg_s"] * recipients["dwt"]
    ).sum() * 12 / 1_000_000

    print(f"\nAggregate expenditure:")
    print(f"  Simulated annual total:  €{total_expenditure:,.2f}M")

    print()
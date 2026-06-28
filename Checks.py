
from __future__ import annotations

import pandas as pd

from src.constants import IMV_FILES, RMI_FILES
from src.exposure_loader import load_all_files

_, imv_dfs = load_all_files(RMI_FILES, IMV_FILES)

checks = [
    (2017, 42),  # Castilla-La Mancha
    (2018, 70),  # Canarias
]

for year, drgn2 in checks:
    df = imv_dfs[year]

    region = df[
        df["drgn2"].eq(drgn2)
    ].copy()

    positive = region[
        region["bsarg_s"].gt(0)
    ].copy()

    n_unique = (
        positive.groupby("idhh")["bsarg_s"]
        .nunique()
    )

    affected_households = n_unique[
        n_unique.gt(1)
    ].index

    print(
        f"\nYear {year}, region {drgn2}, "
        f"affected households: {len(affected_households)}"
    )

    columns = [
        "idhh",
        "idperson",
        "dag",
        "dwt",
        "bsa00_s",
        "bsarg_s",
        "yds",
    ]

    columns = [
        column
        for column in columns
        if column in region.columns
    ]

    print(
        region.loc[
            region["idhh"].isin(affected_households),
            columns,
        ]
        .sort_values(["idhh", "idperson"])
        .to_string(index=False)
    )
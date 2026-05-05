"""
merge.py
========
Merges household-level and person-level UDB DataFrames, validates the result
against the output schema, and exports the final tab-separated file for EUROMOD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.constants import OUTPUT_ENCODING, OUTPUT_MISSING_VALUE, OUTPUT_SEPARATOR, UDB_COLUMN_ORDER
from src.schemas import PersonUdbSchema

logger = logging.getLogger(__name__)


def merge_and_export(
    person_udb: pd.DataFrame,
    household_udb: pd.DataFrame,
    output_path: Path,
    year: int,
) -> pd.DataFrame:
    hh_cols = [c for c in household_udb.columns if c != "year"]
    person_only = [
        c for c in person_udb.columns
        if c not in hh_cols or c == "IDHH"
    ]

    merged = person_udb[person_only].merge(
        household_udb[hh_cols],
        on="IDHH",
        how="left",
        validate="m:1",
    )

    n_before = len(person_udb)
    n_after = len(merged)
    if n_after != n_before:
        raise ValueError(
            f"Year {year}: row count changed in merge ({n_before} → {n_after}). "
            "Check for duplicate IDHH values in household_udb."
        )

    unmatched = merged["drgn1"].isna().sum()
    if unmatched > 0:
        logger.warning(
            "Year %s: %d persons did not match a household record.", year, unmatched
        )

    merged = merged.rename(columns={"IDHH": "idhh", "idperson": "idperson"})

    present = [c for c in UDB_COLUMN_ORDER if c in merged.columns]
    extra   = [c for c in merged.columns if c not in UDB_COLUMN_ORDER and c not in ("year",)]
    if extra:
        logger.debug("Year %s: dropping %d non-UDB columns before export: %s", year, len(extra), extra)

    merged = merged[present]

    for col in present:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    PersonUdbSchema.validate(merged, lazy=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(
        output_path,
        sep=OUTPUT_SEPARATOR,
        index=False,
        na_rep=OUTPUT_MISSING_VALUE,
        encoding=OUTPUT_ENCODING,
    )

    logger.info(
        "Year %s: exported %d persons, %d columns → %s",
        year, len(merged), len(merged.columns), output_path,
    )
    return merged
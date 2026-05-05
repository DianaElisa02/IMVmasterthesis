"""
merge.py
========
Merges household-level and person-level UDB DataFrames, validates the result
against the output schema, and exports the final tab-separated file for EUROMOD.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.constants import OUTPUT_MISSING_VALUE, OUTPUT_SEPARATOR, UDB_COLUMN_ORDER
from src.schemas import PersonUdbSchema

logger = logging.getLogger(__name__)

_STRING_ID_COLS = frozenset({"idhh", "idperson"})


def merge_and_export(
    person_udb: pl.DataFrame,
    household_udb: pl.DataFrame,
    output_path: Path,
    year: int,
) -> pl.DataFrame:
    hh_cols    = [c for c in household_udb.columns if c != "year"]
    person_only = [c for c in person_udb.columns if c not in hh_cols or c == "IDHH"]

    if household_udb.select("IDHH").n_unique() != len(household_udb):
        raise ValueError(f"Year {year}: duplicate IDHH values in household_udb.")

    merged = (
        person_udb.select(person_only)
        .join(household_udb.select(hh_cols), on="IDHH", how="left")
    )

    n_before = len(person_udb)
    n_after  = len(merged)
    if n_after != n_before:
        raise ValueError(
            f"Year {year}: row count changed in merge ({n_before} → {n_after})."
        )

    unmatched = merged["drgn1"].is_null().sum()
    if unmatched > 0:
        logger.warning("Year %s: %d persons did not match a household record.", year, unmatched)

    merged = merged.rename({"IDHH": "idhh"})

    present = [c for c in UDB_COLUMN_ORDER if c in merged.columns]
    extra   = [c for c in merged.columns if c not in UDB_COLUMN_ORDER and c != "year"]
    if extra:
        logger.debug("Year %s: dropping %d non-UDB columns: %s", year, len(extra), extra)

    merged = merged.select(present)

    numeric_cols = [c for c in present if c not in _STRING_ID_COLS]
    merged = merged.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in numeric_cols]
    )

    PersonUdbSchema.validate(merged, lazy=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(
        output_path,
        separator=OUTPUT_SEPARATOR,
        null_value=OUTPUT_MISSING_VALUE,
    )

    logger.info(
        "Year %s: exported %d persons, %d columns → %s",
        year, len(merged), len(merged.columns), output_path,
    )
    return merged

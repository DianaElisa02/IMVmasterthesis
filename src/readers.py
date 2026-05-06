"""
readers.py
==========
Raw ECV file readers for the ECV → EUROMOD UDB conversion pipeline.

Each function reads one ECV section for one year, selecting only the columns
needed for UDB conversion as declared in constants.py. Stata files are read
via pandas (no native Polars .dta reader) and immediately converted to a
Polars DataFrame. No recoding is performed here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.constants import (
    ECV_FILE_PREFIXES,
    TD_COLUMNS,
    TH_COLUMNS,
    TR_COLUMNS,
    TP_COLUMNS,
)

logger = logging.getLogger(__name__)


def _ecv_path(input_dir: Path, file_type: str, year: int) -> Path:
    prefix = ECV_FILE_PREFIXES[file_type]
    return input_dir / f"{prefix}_{year}.dta"

def _read_section(
    path: Path,
    requested_columns: list[str],
    year: int,
    section: str,
) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ECV {section.upper()} file not found: {path}")

    df_pd = pd.read_stata(path, convert_categoricals=False)
    available_upper = {c.upper(): c for c in df_pd.columns}

    selected, missing = [], []
    for col in requested_columns:
        match = available_upper.get(col.upper())
        if match:
            selected.append(match)
        else:
            missing.append(col)

    if missing:
        logger.warning(
            "Year %s | %s: %d columns absent in source, filled with null: %s",
            year, section.upper(), len(missing), missing,
        )

    df_pd = df_pd[[c for c in df_pd.columns if c.upper() in {c.upper() for c in selected}]]
    df_pd.columns = [c.upper() for c in df_pd.columns]
    df = pl.from_pandas(df_pd)

    if missing:
        df = df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(col)
            for col in missing
        ])

    logger.info(
        "Year %s | %s: read %d rows, %d columns (%d null-filled)",
        year, section.upper(), len(df), len(df.columns), len(missing),
    )
    return df

def read_td(input_dir: Path, year: int) -> pl.DataFrame:
    return _read_section(_ecv_path(input_dir, "td", year), TD_COLUMNS, year, "td")


def read_th(input_dir: Path, year: int) -> pl.DataFrame:
    return _read_section(_ecv_path(input_dir, "th", year), TH_COLUMNS, year, "th")


def read_tr(input_dir: Path, year: int) -> pl.DataFrame:
    return _read_section(_ecv_path(input_dir, "tr", year), TR_COLUMNS, year, "tr")


def read_tp(input_dir: Path, year: int) -> pl.DataFrame:
    return _read_section(_ecv_path(input_dir, "tp", year), TP_COLUMNS, year, "tp")

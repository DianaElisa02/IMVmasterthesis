"""
readers.py
==========
Raw ECV file readers for the ECV → EUROMOD UDB conversion pipeline.

Each function reads one ECV section for one year, selecting only the columns
needed for UDB conversion as declared in constants.py. No recoding is performed
here — that is the responsibility of recode.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

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
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"ECV {section.upper()} file not found: {path}")

    available = pd.read_stata(path, convert_categoricals=False).columns.tolist()
    available_upper = {c.upper(): c for c in available}

    selected = []
    missing = []
    for col in requested_columns:
        match = available_upper.get(col.upper())
        if match:
            selected.append(match)
        else:
            missing.append(col)

    if missing:
        logger.warning(
            "Year %s | %s: %d requested columns not found and will be skipped: %s",
            year, section.upper(), len(missing), missing,
        )

    df = pd.read_stata(path, columns=selected, convert_categoricals=False)
    df.columns = [c.upper() for c in df.columns]

    logger.info("Year %s | %s: read %d rows, %d columns", year, section.upper(), len(df), len(df.columns))
    return df


def read_td(input_dir: Path, year: int) -> pd.DataFrame:
    path = _ecv_path(input_dir, "td", year)
    return _read_section(path, TD_COLUMNS, year, "td")


def read_th(input_dir: Path, year: int) -> pd.DataFrame:
    path = _ecv_path(input_dir, "th", year)
    return _read_section(path, TH_COLUMNS, year, "th")


def read_tr(input_dir: Path, year: int) -> pd.DataFrame:
    path = _ecv_path(input_dir, "tr", year)
    return _read_section(path, TR_COLUMNS, year, "tr")


def read_tp(input_dir: Path, year: int) -> pd.DataFrame:
    path = _ecv_path(input_dir, "tp", year)
    return _read_section(path, TP_COLUMNS, year, "tp")
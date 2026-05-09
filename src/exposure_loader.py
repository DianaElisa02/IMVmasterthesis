"""
exposure_loader.py
==================
Loads EUROMOD output files
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = frozenset({
    "idperson", "idhh", "drgn2", "dwt", "dag",
    "bsa00_s", "bsarg_s", "hsize", "yds", "les",
})


def load_euromod_output(path: Path, label: str = "") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"EUROMOD output not found: {path}")

    tag = label or path.name
    logger.info("Loading %s → %s", tag, path)

    df = pd.read_csv(path, sep="\t", low_memory=False, dtype=str)

    for col in df.columns:
        if col in {"idhh", "idperson"}:
            continue
        df[col] = pd.to_numeric(
            df[col].str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{tag}: missing required columns: {sorted(missing)}"
        )

    # Fill NaN in key benefit columns with 0
    for col in ["bsa00_s", "bsarg_s"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            logger.warning("%s: %d nulls in %s — filled with 0", tag, n_null, col)
            df[col] = df[col].fillna(0.0)

    logger.info(
        "%s: loaded %d persons, %d columns",
        tag, len(df), len(df.columns),
    )
    return df


def load_all_files(
    rmi_files: dict[int, Path],
    imv_files: dict[int, Path],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    rmi_dfs: dict[int, pd.DataFrame] = {}
    imv_dfs: dict[int, pd.DataFrame] = {}

    for year in sorted(rmi_files):
        rmi_dfs[year] = load_euromod_output(
            rmi_files[year], label=f"RMI {year}"
        )
        imv_dfs[year] = load_euromod_output(
            imv_files[year], label=f"IMV {year}"
        )

    return rmi_dfs, imv_dfs
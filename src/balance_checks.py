"""
balance_checks.py
=================
Pre-reform balance checks for the IMV DiD analysis.

Computes a Table 1 style balance table comparing pre-reform means of
outcomes and controls across exposure terciles (low / medium / high),
defined on exposure_composite_hybrid at the region level.

"""

from __future__ import annotations

import logging

import polars as pl
import numpy as np

from src.constants import (
    BALANCE_OUTCOMES,
    BALANCE_PRIMARY_SPEC,
)

logger = logging.getLogger(__name__)

def _weighted_mean(df: pl.DataFrame, col: str, weight: str = "weight_hh") -> float:
    sub = df.select([col, weight]).drop_nulls()
    sub = sub.filter(pl.col(weight).gt(0))
    if sub.is_empty():
        return float("nan")
    vals = sub[col].cast(pl.Float64).to_numpy()
    wts  = sub[weight].cast(pl.Float64).to_numpy()
    return float(np.average(vals, weights=wts))


def _assign_terciles(panel: pl.DataFrame) -> pl.DataFrame:
    region_exposure = (
        panel.select(["drgn2", BALANCE_PRIMARY_SPEC])
        .unique(subset=["drgn2"])
        .sort(BALANCE_PRIMARY_SPEC)
    )

    n = len(region_exposure)
    tercile_size = n // 3

    tercile_labels = (
        [1] * tercile_size +
        [2] * tercile_size +
        [3] * (n - 2 * tercile_size)
    )

    region_exposure = region_exposure.with_columns(
        pl.Series("exposure_tercile", tercile_labels, dtype=pl.Int32)
    )

    logger.info(
        "Tercile assignment: %s",
        region_exposure.select(["drgn2", BALANCE_PRIMARY_SPEC, "exposure_tercile"])
        .sort("exposure_tercile")
        .to_pandas()
        .to_string(index=False)
    )

    return panel.join(
        region_exposure.select(["drgn2", "exposure_tercile"]),
        on="drgn2",
        how="left",
    )


def run_balance_checks(panel: pl.DataFrame) -> pl.DataFrame:
    pre = panel.filter(pl.col("post").eq(0.0))
    logger.info("Pre-reform observations for balance checks: %d", len(pre))

    pre = _assign_terciles(pre)

    variables = BALANCE_OUTCOMES
    rows = []

    for var in variables:
        if var not in pre.columns:
            logger.warning("Variable %s not in panel — skipping", var)
            continue

        row = {"variable": var}

        for tercile, label in [(1, "low"), (2, "medium"), (3, "high")]:
            g = pre.filter(pl.col("exposure_tercile").eq(tercile))
            row[f"mean_{label}"] = _weighted_mean(g, var)
            row[f"n_hh_{label}"] = len(g)

        row["mean_full"] = _weighted_mean(pre, var)
        row["n_hh_full"] = len(pre)

        row["diff_high_low"] = row["mean_high"] - row["mean_low"]

        rows.append(row)

    table = pl.DataFrame(rows)
    logger.info("Balance table computed: %d variables", len(table))
    return table
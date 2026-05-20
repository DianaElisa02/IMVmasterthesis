"""
poverty_gap.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

BASE_PATH       = Path("/workspaces/IMVmasterthesis")
ENRICHED_OUTPUT = BASE_PATH / "output" / "analysis_dataset_with_gap.parquet"


def construct_poverty_gap(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Add poverty_gap and poverty_gap_sq columns to the analysis panel.

    Saves the enriched panel to analysis_dataset_with_gap.parquet.

    Parameters
    ----------
    panel : full analysis panel (Polars DataFrame)

    Returns
    -------
    panel with two new columns: poverty_gap, poverty_gap_sq.
    The intermediate column poverty_line is dropped before returning.
    """
    for col in ["income_net_annual", "equiv_income", "weight_hh", "hh_size"]:
        if col not in panel.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                f"income_net_annual=HY020, equiv_income=HX240."
            )

    # ── Equivalised income ────────────────────────────────────────────────────
    panel = panel.with_columns(
        pl.when(
            pl.col("equiv_income").is_not_null() &
            pl.col("equiv_income").gt(0.0) &
            pl.col("income_net_annual").is_not_null()
        )
        .then(pl.col("income_net_annual") / pl.col("equiv_income"))
        .otherwise(pl.lit(None))
        .alias("equivalised_income")
    )

    n_null = panel["equivalised_income"].null_count()
    logger.info(
        "equivalised_income (HY020/HX240): %d obs | %d nulls (%.1f%%)",
        len(panel), n_null, 100 * n_null / len(panel),
    )

    # ── Annual poverty line — 60% of person-weighted median ──────────────────
    years = sorted(panel["year"].unique().to_list())
    poverty_lines: dict[int, float] = {}

    for yr in years:
        yr_df = (
            panel.filter(pl.col("year").eq(yr))
            .select(["equivalised_income", "weight_hh", "hh_size"])
            .drop_nulls()
            .filter(pl.col("weight_hh").gt(0))   # exclude INE zero-weight obs
        )
        if len(yr_df) == 0:
            logger.warning("Year %d: no valid observations for poverty line", yr)
            continue

        equiv_vals     = yr_df["equivalised_income"].to_numpy()
        # Person-level weight: household weight × household size
        person_weights = yr_df["weight_hh"].to_numpy() * yr_df["hh_size"].to_numpy()

        # Weighted median via sorted cumulative weights
        sort_idx        = np.argsort(equiv_vals)
        sorted_vals     = equiv_vals[sort_idx]
        cum_weights     = np.cumsum(person_weights[sort_idx])
        median_idx      = np.searchsorted(cum_weights, cum_weights[-1] / 2.0)
        weighted_median = float(sorted_vals[min(median_idx, len(sorted_vals) - 1)])

        poverty_lines[yr] = 0.60 * weighted_median
        logger.info(
            "Year %d: weighted median = €%.0f | poverty line (60%%) = €%.0f",
            yr, weighted_median, poverty_lines[yr],
        )

    # ── Map poverty lines and compute gap ─────────────────────────────────────
    poverty_line_map = pl.DataFrame({
        "year":         list(poverty_lines.keys()),
        "poverty_line": list(poverty_lines.values()),
    }).with_columns(pl.col("year").cast(panel["year"].dtype))

    panel = panel.join(poverty_line_map, on="year", how="left")

    panel = panel.with_columns(
        pl.when(
            pl.col("equivalised_income").is_not_null() &
            pl.col("poverty_line").is_not_null()
        )
        .then(
            pl.max_horizontal(
                pl.lit(0.0),
                (pl.col("poverty_line") - pl.col("equivalised_income")) /
                pl.col("poverty_line")
            )
        )
        .otherwise(pl.lit(None))
        .alias("poverty_gap")
    ).with_columns(
        (pl.col("poverty_gap") ** 2).alias("poverty_gap_sq")
    ).drop("poverty_line")

    # ── Descriptive summary ───────────────────────────────────────────────────
    eq_mean = float(panel["equivalised_income"].drop_nulls().mean())
    logger.info(
        "Mean equivalised income: €%.0f (expected ~€18,000–€22,000 for Spain 2017–2025)",
        eq_mean,
    )
    for outcome in ["poverty_gap", "poverty_gap_sq"]:
        n_pos  = int((panel[outcome].drop_nulls() > 0).sum())
        n_null_o = int(panel[outcome].null_count())
        logger.info(
            "%s: gap>0 = %d (%.1f%%) | nulls = %d",
            outcome, n_pos, 100 * n_pos / len(panel), n_null_o,
        )

    # ── Save enriched panel ───────────────────────────────────────────────────
    panel.write_parquet(ENRICHED_OUTPUT)
    logger.info("Enriched panel saved: %s", ENRICHED_OUTPUT)

    return panel
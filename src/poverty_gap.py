"""
poverty_gap.py
==============

Constructs poverty_gap, poverty_gap_sq, and weight_person columns for the
IMV analysis panel.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_SRC_DIR        = Path(__file__).resolve().parent
_PROJECT_ROOT   = _SRC_DIR.parent
ENRICHED_OUTPUT = _PROJECT_ROOT / "output" / "analysis_dataset_with_gap.parquet"
print(f"[poverty_gap] ENRICHED_OUTPUT = {ENRICHED_OUTPUT}")
POVERTY_LINE_SHARE = 0.60


def construct_poverty_gap(panel: pl.DataFrame) -> pl.DataFrame:
    for col in ["income_net_annual", "equiv_income", "weight_hh", "hh_size"]:
        if col not in panel.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                f"Expected: income_net_annual=HY020, equiv_income=HX240 "
                f"(OECD-modified equivalence scale, NOT already-equivalised "
                f"income HX090)."
            )

    years = sorted(panel["year"].unique().to_list())
    if 2020 in years:
        logger.warning(
            "Year 2020 detected in panel passed to construct_poverty_gap. "
            "Per the identification strategy, 2020 should be excluded before "
            "this function is called (mid-year IMV introduction + COVID "
            "confounding). Poverty lines will be computed for 2020 but "
            "these observations should not enter the DiD estimation."
        )

    panel = panel.with_columns(
        (pl.col("weight_hh") * pl.col("hh_size")).alias("weight_person")
    )
    logger.info("weight_person column added (weight_hh × hh_size)")

    panel = panel.with_columns(
        pl.when(
            pl.col("equiv_income").is_not_null() &
            pl.col("equiv_income").gt(0.0) &
            pl.col("income_net_annual").is_not_null()
        )
        .then(
            pl.max_horizontal(
                pl.col("income_net_annual"),
                pl.lit(0.0)
            ) / pl.col("equiv_income")
        )
        .otherwise(pl.lit(None))
        .alias("equivalised_income")
    )

    n_null = panel["equivalised_income"].null_count()
    logger.info(
        "equivalised_income (HY020 / HX240): %d obs | %d nulls (%.1f%%)",
        len(panel), n_null, 100 * n_null / len(panel),
    )

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

        equiv_vals = yr_df["equivalised_income"].to_numpy()
        person_weights = yr_df["weight_hh"].to_numpy() * yr_df["hh_size"].to_numpy()

        sort_idx        = np.argsort(equiv_vals)
        sorted_vals     = equiv_vals[sort_idx]
        cum_weights     = np.cumsum(person_weights[sort_idx])
        median_idx      = np.searchsorted(
            cum_weights, cum_weights[-1] / 2.0, side="left"   # FIX 3
        )
        weighted_median = float(sorted_vals[min(median_idx, len(sorted_vals) - 1)])

        poverty_lines[yr] = POVERTY_LINE_SHARE * weighted_median
        logger.info(
            "Year %d: person-weighted median = €%.0f | "
            "poverty line (%.0f%%) = €%.0f",
            yr, weighted_median,
            POVERTY_LINE_SHARE * 100,
            poverty_lines[yr],
        )

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
    )

    panel = panel.with_columns(
        pl.when(pl.col("poverty_gap").is_not_null())
        .then(pl.col("poverty_gap") ** 2)
        .otherwise(pl.lit(None))
        .alias("poverty_gap_sq")
    )

    panel = panel.drop("poverty_line")

    eq_mean = float(panel["equivalised_income"].drop_nulls().mean())
    logger.info(
        "Mean equivalised income: €%.0f "
        "(expected ~€18,000–€22,000 for Spain 2017–2025)",
        eq_mean,
    )
    for outcome in ["poverty_gap", "poverty_gap_sq"]:
        n_pos    = int((panel[outcome].drop_nulls() > 0).sum())
        n_null_o = int(panel[outcome].null_count())
        logger.info(
            "%s: gap>0 = %d (%.1f%%) | nulls = %d",
            outcome, n_pos, 100 * n_pos / len(panel), n_null_o,
        )

    wp_mean = float(panel["weight_person"].drop_nulls().mean())
    logger.info("Mean weight_person: %.2f", wp_mean)

    panel.write_parquet(ENRICHED_OUTPUT)
    logger.info("Enriched panel saved: %s", ENRICHED_OUTPUT)

    return panel
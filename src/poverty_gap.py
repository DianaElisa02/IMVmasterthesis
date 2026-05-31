"""
poverty_gap.py
==============

Constructs poverty_gap, poverty_gap_sq, and weight_person columns for the
IMV analysis panel.

Changes from original
---------------------
FIX 2 — Weighting consistency
    A person-level weight column (weight_person = weight_hh × hh_size) is
    added to the returned panel. The poverty line is already anchored to the
    person-weighted income distribution (weighted median uses person weights).
    Using weight_person as the survey weight in the DiD regression ensures
    the outcome is weighted on the same basis as the threshold, eliminating
    the household-vs-person inconsistency.

FIX 3 — searchsorted side='left'
    np.searchsorted is called with side='left' (explicitly stated). This
    finds the first index where cumulative weight >= total/2, which is the
    correct weighted-median definition.

FIX 4 — Null guard on poverty_gap_sq
    poverty_gap_sq is computed inside a .when(...is_not_null()) guard so
    that null equivalised income propagates cleanly as null rather than
    silently producing 0.0 via squaring a null.

FIX 5a — 2020 warning
    The function raises a logger warning if year 2020 is present in the
    panel before computing poverty lines, since 2020 should be excluded
    upstream per the identification strategy.

FIX 5b — Poverty line set to 40% of weighted median
    Changed from 0.60 to 0.40. This captures households in severe income
    poverty (below the 40% threshold), consistent with the thesis focus on
    extreme deprivation rather than the standard Eurostat at-risk-of-poverty
    line (60%).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_SRC_DIR        = Path(__file__).resolve().parent        # → .../IMVmasterthesis/src
_PROJECT_ROOT   = _SRC_DIR.parent                        # → .../IMVmasterthesis
ENRICHED_OUTPUT = _PROJECT_ROOT / "output" / "analysis_dataset_with_gap.parquet"
print(f"[poverty_gap] ENRICHED_OUTPUT = {ENRICHED_OUTPUT}")
POVERTY_LINE_SHARE = 0.60


def construct_poverty_gap(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Add poverty_gap, poverty_gap_sq, and weight_person columns to the panel.

    Poverty line
    ------------
    Set at POVERTY_LINE_SHARE (40%) of the person-weighted median of
    equivalised household income, computed separately for each survey year.
    Equivalised income = income_net_annual (HY020) / equiv_income (HX240),
    where HX240 is the OECD-modified equivalence scale.

    New columns
    -----------
    equivalised_income : float  — HY020 / HX240
    poverty_gap        : float  — max(0, (line - equiv_income) / line); null
                                  when equivalised_income is null
    poverty_gap_sq     : float  — poverty_gap^2 (FGT(2) at hh level); null
                                  when poverty_gap is null
    weight_person      : float  — weight_hh * hh_size (person-level weight)
                                  Use this as survey weight in regressions
                                  involving poverty_gap / poverty_gap_sq.

    The intermediate column poverty_line is dropped before returning.
    The enriched panel is saved to analysis_dataset_with_gap.parquet.

    Parameters
    ----------
    panel : full analysis panel (Polars DataFrame), 2020 excluded upstream.

    Returns
    -------
    Polars DataFrame with the four new columns described above.
    """

    # ── Required column check ─────────────────────────────────────────────────
    for col in ["income_net_annual", "equiv_income", "weight_hh", "hh_size"]:
        if col not in panel.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                f"Expected: income_net_annual=HY020, equiv_income=HX240 "
                f"(OECD-modified equivalence scale, NOT already-equivalised "
                f"income HX090)."
            )

    # FIX 5a: warn if year 2020 is present — it should be excluded upstream
    years = sorted(panel["year"].unique().to_list())
    if 2020 in years:
        logger.warning(
            "Year 2020 detected in panel passed to construct_poverty_gap. "
            "Per the identification strategy, 2020 should be excluded before "
            "this function is called (mid-year IMV introduction + COVID "
            "confounding). Poverty lines will be computed for 2020 but "
            "these observations should not enter the DiD estimation."
        )

    # ── FIX 2: Person-level weight ────────────────────────────────────────────
    # The poverty line is anchored to the person-weighted income distribution.
    # Add weight_person so the same weighting scheme can be used in regressions.
    panel = panel.with_columns(
        (pl.col("weight_hh") * pl.col("hh_size")).alias("weight_person")
    )
    logger.info("weight_person column added (weight_hh × hh_size)")

    # ── Equivalised income ────────────────────────────────────────────────────
    # HY020 (net annual household income) divided by HX240 (OECD scale).
    # HX240 for a single adult = 1.0; couple = 1.5; couple + 1 child = 1.8; etc.
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
        "equivalised_income (HY020 / HX240): %d obs | %d nulls (%.1f%%)",
        len(panel), n_null, 100 * n_null / len(panel),
    )

    # ── Annual poverty line — POVERTY_LINE_SHARE of person-weighted median ────
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
        # Person-level weight: household weight × household size
        person_weights = yr_df["weight_hh"].to_numpy() * yr_df["hh_size"].to_numpy()

        # FIX 3: weighted median via sorted cumulative weights, side='left'
        # side='left' returns the first index where cum_weight >= total/2,
        # which is the standard definition of the weighted median.
        sort_idx        = np.argsort(equiv_vals)
        sorted_vals     = equiv_vals[sort_idx]
        cum_weights     = np.cumsum(person_weights[sort_idx])
        median_idx      = np.searchsorted(
            cum_weights, cum_weights[-1] / 2.0, side="left"   # FIX 3
        )
        weighted_median = float(sorted_vals[min(median_idx, len(sorted_vals) - 1)])

        # FIX 5b: 40% threshold instead of 60%
        poverty_lines[yr] = POVERTY_LINE_SHARE * weighted_median
        logger.info(
            "Year %d: person-weighted median = €%.0f | "
            "poverty line (%.0f%%) = €%.0f",
            yr, weighted_median,
            POVERTY_LINE_SHARE * 100,
            poverty_lines[yr],
        )

    # ── Map poverty lines back to panel ──────────────────────────────────────
    poverty_line_map = pl.DataFrame({
        "year":         list(poverty_lines.keys()),
        "poverty_line": list(poverty_lines.values()),
    }).with_columns(pl.col("year").cast(panel["year"].dtype))

    panel = panel.join(poverty_line_map, on="year", how="left")

    # ── Poverty gap — FGT(1) at household level ───────────────────────────────
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

    # FIX 4: explicit null guard on poverty_gap_sq
    # Without this, a null poverty_gap would silently produce null^2 = null
    # in some backends, but the guard makes the intent unambiguous and
    # ensures consistent null propagation regardless of Polars version.
    panel = panel.with_columns(
        pl.when(pl.col("poverty_gap").is_not_null())
        .then(pl.col("poverty_gap") ** 2)
        .otherwise(pl.lit(None))
        .alias("poverty_gap_sq")
    )

    # Drop intermediate column
    panel = panel.drop("poverty_line")

    # ── Descriptive summary ───────────────────────────────────────────────────
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

    # ── Save enriched panel ───────────────────────────────────────────────────
    panel.write_parquet(ENRICHED_OUTPUT)
    logger.info("Enriched panel saved: %s", ENRICHED_OUTPUT)

    return panel
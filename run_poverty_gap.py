"""
run_poverty_gap.py
==================
Constructs poverty gap outcomes and runs the full DiD pipeline on them.

Motivation
----------
The primary outcomes matdep and poverty are binary indicators with
low base rates (~5% and ~20% respectively). The IMV is an income
top-up transfer — it may reduce the depth of poverty without pushing
households above the binary threshold. The poverty gap (FGT-1) and
poverty gap squared (FGT-2) are continuous outcomes that capture
how far below the poverty line households fall, and are therefore
more sensitive to the kind of partial income improvement the IMV
provides.

This is the standard outcome in the closest existing literature:
  - Hernández, Picos & Riscado (2020) — JRC paper on Spanish RMI
  - Almeida, De Poli & Hernández (2025) — EU minimum income review

Construction
------------
equivalised_income_h = HY020_h / HX240_h
    where HY020 = net disposable household income (already in parquet)
          HX240 = OECD modified equivalence scale (already in parquet)

poverty_line_t = 0.60 × weighted median of equivalised_income
    computed annually, using survey weights (weight_hh)
    weighted at person level (INE convention: median of person distribution)

poverty_gap_h = max(0, poverty_line_t - equivalised_income_h) / poverty_line_t
    continuous [0, 1] — 0 means at or above poverty line
    interpreted as: fraction of poverty line by which household falls short

poverty_gap_sq_h = poverty_gap_h^2
    FGT-2: weights poorest households quadratically more heavily

Key decisions
-------------
1. Use HY020 / HX240 rather than vhRentaa — vhRentaa is not in the
   current parquet. HY020 / HX240 is the standard equivalisation and
   matches INE's construction of vhRentaa closely.
2. Poverty line computed on the FULL national sample (all regions,
   all observation types) weighted at person level — consistent with
   INE methodology.
3. Poverty line computed separately per year — allows the line to
   move with the income distribution over time, as per INE convention.
4. Poverty gap is a continuous outcome — estimated via OLS (not LPM),
   coefficients interpreted as fractional change in poverty gap.
5. Same DiD specification, inference, and robustness structure as
   matdep and poverty.

Parallel trends check
---------------------
The poverty gap must pass the placebo test before being used causally.
This script runs the placebo test on the poverty gap before the DiD.
If placebo fails, the poverty gap is treated as an appendix outcome only.

Usage
-----
  python run_poverty_gap.py
"""

from __future__ import annotations

import logging
import io
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats as scipy_stats

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EVENT_STUDY_REFERENCE_YEAR,
    EVENT_STUDY_YEARS,
    EXPOSURE_SPECS,
    PLACEBO_FAKE_TREATMENT_YEAR,
    PLACEBO_REFERENCE_YEAR,
    PLACEBO_YEARS,
    REGION_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "poverty_gap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPEC = EXPOSURE_SPECS[0]
POVERTY_GAP_OUTCOMES = ["poverty_gap", "poverty_gap_sq"]

N_CLUSTERS = 15


# =============================================================================
# STEP 1 — CONSTRUCT POVERTY GAP OUTCOMES
# =============================================================================

def construct_poverty_gap(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Add poverty_gap and poverty_gap_sq to the panel.

    Construction
    ------------
    equiv_income in the parquet (column name 'equiv_income') = HX240,
    the OECD modified equivalence scale (consumption units), NOT income.
    It must be used as a denominator to equivalise income:

        equivalised_income = HY020 / HX240
                           = income_net_annual / equiv_income

    Poverty line = 0.60 × weighted median of equivalised_income,
    computed annually, weighted at person level (INE convention).

    poverty_gap     = max(0, poverty_line - equivalised_income) / poverty_line
    poverty_gap_sq  = poverty_gap^2  (FGT-2, weights poorest more)

    Parameters
    ----------
    panel : full analysis panel from analysis_dataset.parquet

    Returns
    -------
    panel with two additional columns: poverty_gap, poverty_gap_sq
    """
    # ── Validate required columns ─────────────────────────────────────────────
    for col in ["income_net_annual", "equiv_income"]:
        if col not in panel.columns:
            raise ValueError(
                f"Column '{col}' not found in panel. "
                f"income_net_annual = HY020, equiv_income = HX240 (scale)."
            )

    # ── Step 1: Construct equivalised income = HY020 / HX240 ─────────────────
    # equiv_income (HX240) is the OECD equivalence scale, e.g.:
    #   single adult    → 1.0
    #   two adults      → 1.5
    #   two adults + 2 children → 1.5 + 0.3 + 0.3 = 2.1
    # Dividing household income by this scale gives income per consumption unit.
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
        "equivalised_income constructed (HY020/HX240): %d obs, %d nulls (%.1f%%)",
        len(panel), n_null, 100 * n_null / len(panel),
    )

    # ── Step 2: Annual poverty line ───────────────────────────────────────────
    # 60% of national weighted median of equivalised_income, per year.
    # Person-level weighting: hh_size × weight_hh (INE convention).
    years = sorted(panel["year"].unique().to_list())
    poverty_lines: dict[int, float] = {}

    for yr in years:
        yr_df = (
            panel.filter(pl.col("year").eq(yr))
            .select(["equivalised_income", "weight_hh", "hh_size"])
            .drop_nulls()
        )

        if len(yr_df) == 0:
            logger.warning("Year %d: no valid observations for poverty line", yr)
            continue

        equiv_vals    = yr_df["equivalised_income"].to_numpy()
        person_weights = (
            yr_df["weight_hh"].to_numpy() * yr_df["hh_size"].to_numpy()
        )

        # Weighted median via sorted cumulative weights
        sort_idx        = np.argsort(equiv_vals)
        sorted_vals     = equiv_vals[sort_idx]
        sorted_weights  = person_weights[sort_idx]
        cum_weights     = np.cumsum(sorted_weights)
        total_weight    = cum_weights[-1]
        median_idx      = np.searchsorted(cum_weights, total_weight / 2.0)
        weighted_median = float(sorted_vals[min(median_idx, len(sorted_vals) - 1)])

        poverty_line = 0.60 * weighted_median
        poverty_lines[yr] = poverty_line

        logger.info(
            "Year %d: weighted median equivalised income = €%.0f | "
            "poverty line (60%%) = €%.0f",
            yr, weighted_median, poverty_line,
        )

    # ── Step 3: Map poverty lines and construct gap ───────────────────────────
    poverty_line_map = pl.DataFrame({
        "year":         list(poverty_lines.keys()),
        "poverty_line": list(poverty_lines.values()),
    }).with_columns(pl.col("year").cast(pl.Int64))

    panel = panel.with_columns(pl.col("year").cast(pl.Int64))
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
        (pl.col("poverty_gap") ** 2).alias("poverty_gap_sq")
    )

    # Descriptive summary
    for yr in sorted(poverty_lines.keys()):
        yr_df = panel.filter(pl.col("year").eq(yr))
        pg    = yr_df["poverty_gap"].drop_nulls()
        pct_poor = float((pg.gt(0)).mean()) * 100
        mean_gap = float(pg.filter(pg.gt(0)).mean()) if pct_poor > 0 else 0.0
        logger.info(
            "Year %d: poverty gap > 0: %.1f%% of hh | mean gap (poor): %.3f",
            yr, pct_poor, mean_gap,
        )

    panel = panel.drop("poverty_line")
    return panel


# =============================================================================
# STEP 2 — SHARED ESTIMATION UTILITIES
# =============================================================================

def _extract_wbt_pvalue(wbt, col: str) -> float:
    P_VALUE_KEYS = ["p-value", "p_value", "pvalue", "Pr(>|t|)"]
    if isinstance(wbt, pd.DataFrame):
        for key in P_VALUE_KEYS:
            if key in wbt.columns:
                return float(wbt[key].iloc[0])
    elif isinstance(wbt, dict):
        for key in P_VALUE_KEYS:
            if key in wbt:
                raw = wbt[key]
                return float(raw.iloc[0]) if hasattr(raw, "iloc") else float(raw)
    elif isinstance(wbt, (float, int, np.floating, np.integer)):
        return float(wbt)
    raise ValueError(f"Unrecognised WCB return type for '{col}': {type(wbt)}")


def _estimate(
    df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    year_dummy_cols: list[str],
    region_cols: list[str],
    controls: list[str],
    cluster_col: str = "drgn2",
    weight_col: str = "weight_hh",
    wcb_param: str | None = None,
) -> tuple[object, dict[str, float]]:
    """
    Core WLS + WCB estimation. Returns (result, wbt_results).
    """
    regressors = (
        [treatment_col] + year_dummy_cols + region_cols + controls
    )
    regressors = [c for c in regressors if c in df.columns]

    X = sm.add_constant(df[regressors])
    y = df[outcome]
    w = df[weight_col]

    rank   = np.linalg.matrix_rank(X.values)
    n_cols = X.shape[1]
    if rank < n_cols:
        raise ValueError(
            f"Rank deficiency: rank={rank}, n_cols={n_cols}"
        )

    model  = sm.WLS(y, X, weights=w)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]},
    )

    # WCB
    wbt_results: dict[str, float] = {}
    param = wcb_param or treatment_col
    if param not in result.params.index:
        return result, wbt_results

    try:
        from wildboottest.wildboottest import wildboottest
        _trap = io.StringIO()
        with contextlib.redirect_stdout(_trap):
            wbt = wildboottest(
                model,
                cluster=df[cluster_col].values,
                param=param,
                B=9999,
                weights_type="webb",
                seed=42,
            )
        wbt_results[param] = _extract_wbt_pvalue(wbt, param)
    except Exception as e:
        logger.warning("WCB failed for %s: %s", param, e)

    return result, wbt_results


def _build_region_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    region_dummies = pd.get_dummies(
        df["drgn2"], prefix="reg", drop_first=True
    ).astype(float)
    df = pd.concat([df, region_dummies], axis=1)
    return df, region_dummies.columns.tolist()


def _t_crit(n_clusters: int) -> float:
    return float(scipy_stats.t.ppf(0.975, df=n_clusters - 1))


# =============================================================================
# STEP 3 — PLACEBO TEST (validation before DiD)
# =============================================================================

def run_placebo_poverty_gap(panel: pl.DataFrame) -> pd.DataFrame:
    """
    Placebo test for poverty gap outcomes using pre-reform years only.
    Fake treatment at 2019, reference 2018.
    """
    logger.info("--- Placebo test: poverty gap outcomes ---")

    placebo = panel.filter(pl.col("year").is_in(PLACEBO_YEARS))
    placebo = placebo.with_columns(
        pl.when(pl.col("year").eq(PLACEBO_FAKE_TREATMENT_YEAR))
        .then(pl.lit(1.0))
        .otherwise(pl.lit(0.0))
        .alias("post_fake")
    )
    placebo = placebo.with_columns(
        (pl.col("post_fake") * pl.col(PRIMARY_SPEC))
        .alias("post_fake_x_exposure")
    )

    controls = [c for c in BALANCE_CONTROLS if c in placebo.columns]
    results  = []

    for outcome in POVERTY_GAP_OUTCOMES:
        if outcome not in placebo.columns:
            logger.warning("Outcome %s not in panel — skipping", outcome)
            continue

        keep = (
            ["household_id", "drgn2", "year", outcome, "weight_hh",
             "post_fake_x_exposure"]
            + [f"yr_{yr}" for yr in PLACEBO_YEARS
               if yr != PLACEBO_REFERENCE_YEAR]
            + controls
        )

        df = placebo.select(
            [c for c in keep if c in placebo.columns]
        ).to_pandas().dropna(subset=[outcome, "post_fake_x_exposure"])

        # Year dummies — 2018 omitted
        for yr in PLACEBO_YEARS:
            if yr != PLACEBO_REFERENCE_YEAR:
                df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
        year_dummy_cols = [
            f"yr_{yr}" for yr in PLACEBO_YEARS
            if yr != PLACEBO_REFERENCE_YEAR
        ]

        df, region_cols = _build_region_dummies(df)
        ctrl = [c for c in controls if c in df.columns]

        result, wbt = _estimate(
            df, outcome, "post_fake_x_exposure",
            year_dummy_cols, region_cols, ctrl,
            wcb_param="post_fake_x_exposure",
        )

        n_clusters = df["drgn2"].nunique()
        tc = _t_crit(n_clusters)
        coef  = result.params.get("post_fake_x_exposure", np.nan)
        se    = result.bse.get("post_fake_x_exposure", np.nan)
        pval  = result.pvalues.get("post_fake_x_exposure", np.nan)
        p_wbt = wbt.get("post_fake_x_exposure", np.nan)

        verdict = (
            "PASS" if (not np.isnan(p_wbt) and p_wbt > 0.1)
            else "WARNING"
        )

        logger.info(
            "Placebo %s: β=%.4f SE=%.4f p_cluster=%.4f p_WCB=%.4f → %s",
            outcome, coef, se, pval,
            p_wbt if not np.isnan(p_wbt) else -99,
            verdict,
        )

        results.append({
            "outcome":      outcome,
            "coef":         coef,
            "se":           se,
            "ci_low":       coef - tc * se,
            "ci_high":      coef + tc * se,
            "pval_cluster": pval,
            "pval_wbt":     p_wbt,
            "n_obs":        int(result.nobs),
            "n_clusters":   n_clusters,
            "verdict":      verdict,
        })

        print(f"\n=== Placebo — {outcome} ===")
        print(f"  β         = {coef:.4f}")
        print(f"  SE        = {se:.4f}")
        print(f"  CI        = [{coef - tc*se:.4f}, {coef + tc*se:.4f}]")
        print(f"  p_cluster = {pval:.4f}")
        print(f"  p_WCB     = {p_wbt:.4f}" if not np.isnan(p_wbt)
              else "  p_WCB     = unavailable")
        print(f"  → {verdict}")

    return pd.DataFrame(results)


# =============================================================================
# STEP 4 — EVENT STUDY
# =============================================================================

def run_event_study_poverty_gap(panel: pl.DataFrame) -> None:
    """
    Event study for poverty gap outcomes — year × exposure interactions.
    """
    logger.info("--- Event study: poverty gap outcomes ---")

    es_panel = panel.filter(pl.col("year").ne(2020))
    for year in EVENT_STUDY_YEARS:
        es_panel = es_panel.with_columns(
            pl.when(pl.col("year").eq(year))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias(f"yr_{year}")
        )
        es_panel = es_panel.with_columns(
            (pl.col(f"yr_{year}") * pl.col(PRIMARY_SPEC))
            .alias(f"yr_{year}_x_exposure")
        )

    interaction_cols = [f"yr_{y}_x_exposure" for y in EVENT_STUDY_YEARS]
    year_dummy_cols  = [f"yr_{y}" for y in EVENT_STUDY_YEARS]
    controls = [c for c in BALANCE_CONTROLS if c in es_panel.columns]

    tc = _t_crit(N_CLUSTERS)

    for outcome in POVERTY_GAP_OUTCOMES:
        if outcome not in es_panel.columns:
            continue

        keep = (
            ["household_id", "drgn2", "year", outcome, "weight_hh"]
            + interaction_cols + year_dummy_cols + controls
        )
        df = (
            es_panel.select([c for c in keep if c in es_panel.columns])
            .to_pandas()
            .dropna(subset=[outcome] + interaction_cols)
            .reset_index(drop=True)
        )

        df, region_cols = _build_region_dummies(df)
        ctrl = [c for c in controls if c in df.columns]

        regressors = interaction_cols + year_dummy_cols + region_cols + ctrl
        regressors = [c for c in regressors if c in df.columns]

        X = sm.add_constant(df[regressors])
        y = df[outcome]
        w = df["weight_hh"]

        model  = sm.WLS(y, X, weights=w)
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df["drgn2"]},
        )

        logger.info(
            "Event study %s: R²=%.4f N=%d", outcome, result.rsquared, int(result.nobs)
        )

        # WCB for all interactions
        wbt_all: dict[str, float] = {}
        try:
            from wildboottest.wildboottest import wildboottest
            for col in interaction_cols:
                if col not in result.params.index:
                    continue
                _trap = io.StringIO()
                with contextlib.redirect_stdout(_trap):
                    wbt = wildboottest(
                        model,
                        cluster=df["drgn2"].values,
                        param=col,
                        B=9999,
                        weights_type="webb",
                        seed=42,
                    )
                try:
                    wbt_all[col] = _extract_wbt_pvalue(wbt, col)
                    logger.info("WCB %s %s: p=%.4f", outcome, col, wbt_all[col])
                except Exception as e:
                    logger.warning("WCB extraction failed %s: %s", col, e)
        except Exception as e:
            logger.warning("WCB failed: %s", e)

        # Build coefficient table
        rows = []
        for year in EVENT_STUDY_YEARS:
            col  = f"yr_{year}_x_exposure"
            coef = result.params.get(col, np.nan)
            se   = result.bse.get(col, np.nan)
            rows.append({
                "year":     year,
                "coef":     coef,
                "se":       se,
                "ci_low":   coef - tc * se,
                "ci_high":  coef + tc * se,
                "pval":     result.pvalues.get(col, np.nan),
                "pval_wbt": wbt_all.get(col, np.nan),
            })
        rows.append({
            "year": EVENT_STUDY_REFERENCE_YEAR,
            "coef": 0.0, "se": 0.0,
            "ci_low": 0.0, "ci_high": 0.0,
            "pval": np.nan, "pval_wbt": np.nan,
        })

        coef_table = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

        print(f"\n=== Event study — {outcome} ===")
        print(coef_table.to_string(index=False, float_format="{:.4f}".format))

        # Pre-trend WCB summary
        pre_wbt = {k: v for k, v in wbt_all.items()
                   if "2017" in k or "2018" in k}
        if pre_wbt:
            print("\nPre-trend WCB:")
            for col, p in pre_wbt.items():
                print(f"  {col}: p={p:.4f} {'✓' if p > 0.1 else '⚠ WARNING'}")

        coef_table.to_csv(
            OUTPUT_DIR / f"event_study_{outcome}.csv", index=False
        )


# =============================================================================
# STEP 5 — BASELINE DiD
# =============================================================================

def run_did_poverty_gap(
    panel: pl.DataFrame,
    post_years: list[int],
    label: str,
) -> pd.DataFrame:
    """
    Baseline DiD for poverty gap outcomes — all 5 exposure specs.
    """
    pre_years  = [2017, 2018, 2019]
    keep_years = pre_years + post_years

    did = panel.filter(pl.col("year").is_in(keep_years))
    did = did.with_columns(
        pl.when(pl.col("year").is_in(post_years))
        .then(pl.lit(1.0))
        .when(pl.col("year").is_in(pre_years))
        .then(pl.lit(0.0))
        .otherwise(pl.lit(None))
        .alias("post_did")
    )

    for spec in EXPOSURE_SPECS:
        if spec in did.columns:
            did = did.with_columns(
                (pl.col("post_did") * pl.col(spec))
                .alias(f"post_x_{spec}_did")
            )

    controls = [c for c in BALANCE_CONTROLS if c in did.columns]
    tc       = _t_crit(N_CLUSTERS)
    rows     = []

    for outcome in POVERTY_GAP_OUTCOMES:
        if outcome not in did.columns:
            continue

        for spec in EXPOSURE_SPECS:
            interaction_col = f"post_x_{spec}_did"
            if interaction_col not in did.columns:
                continue

            keep = (
                ["household_id", "drgn2", "year", outcome,
                 "weight_hh", "post_did", interaction_col]
                + controls
            )
            df = (
                did.select([c for c in keep if c in did.columns])
                .to_pandas()
                .dropna(subset=[outcome, interaction_col])
                .reset_index(drop=True)
            )

            # Year dummies — 2019 reference
            years_in_sample = sorted(df["year"].unique().tolist())
            year_dummy_cols = []
            for yr in years_in_sample:
                if yr != 2019:
                    df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
                    year_dummy_cols.append(f"yr_{yr}")

            df, region_cols = _build_region_dummies(df)
            ctrl = [c for c in controls if c in df.columns]

            try:
                result, wbt = _estimate(
                    df, outcome, interaction_col,
                    year_dummy_cols, region_cols, ctrl,
                    wcb_param=interaction_col,
                )
            except Exception as e:
                logger.error("DiD failed %s × %s: %s", outcome, spec, e)
                continue

            n_clusters = df["drgn2"].nunique()
            coef  = result.params.get(interaction_col, np.nan)
            se    = result.bse.get(interaction_col, np.nan)
            pval  = result.pvalues.get(interaction_col, np.nan)
            p_wbt = wbt.get(interaction_col, np.nan)

            logger.info(
                "DiD [%s] %s × %s: β=%.4f p_WCB=%.4f",
                label, outcome, spec, coef,
                p_wbt if not np.isnan(p_wbt) else -99,
            )

            rows.append({
                "label":        label,
                "outcome":      outcome,
                "exposure_spec": spec,
                "coef":         coef,
                "se":           se,
                "ci_low":       coef - tc * se,
                "ci_high":      coef + tc * se,
                "pval_cluster": pval,
                "pval_wbt":     p_wbt,
                "n_obs":        int(result.nobs),
                "n_clusters":   n_clusters,
                "r_squared":    result.rsquared,
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — run_poverty_gap.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    # Check required columns
    # income_net_annual = HY020 (total net disposable income)
    # equiv_income      = HX240 (OECD equivalence scale — denominator only)
    # equivalised_income is constructed here as income_net_annual / equiv_income
    required = ["income_net_annual", "equiv_income", "weight_hh", "hh_size",
                "year", "drgn2", PRIMARY_SPEC]
    missing  = [c for c in required if c not in panel.columns]
    if missing:
        logger.error(
            "Missing required columns: %s. "
            "Check that consumption_units (HX240) and income_net_annual "
            "(HY020) are in the analysis_dataset.parquet.",
            missing,
        )
        raise SystemExit(1)

    # ── Step 1: Construct poverty gap ─────────────────────────────────────────
    logger.info("Step 1: Constructing poverty gap outcomes")
    panel = construct_poverty_gap(panel)

    # Save poverty lines to output for reference
    yr_lines = (
        panel.select(["year", "poverty_line"] if "poverty_line" in panel.columns
                     else ["year"])
        .unique("year")
        .sort("year")
    )

    # Descriptive check
    for outcome in POVERTY_GAP_OUTCOMES:
        if outcome in panel.columns:
            n_pos  = int((panel[outcome] > 0).sum())
            n_zero = int((panel[outcome] == 0).sum())
            n_null = int(panel[outcome].null_count())
            logger.info(
                "Descriptive %s: gap>0=%d (%.1f%%) | gap=0=%d | null=%d",
                outcome, n_pos, 100*n_pos/len(panel), n_zero, n_null,
            )

    # Validate equivalised_income looks plausible
    eq_mean = panel["equivalised_income"].drop_nulls().mean()
    logger.info(
        "equivalised_income (HY020/HX240): mean=€%.0f — "
        "expected ~€12,000-€16,000 for Spain 2017-2025", eq_mean,
    )

    # ── Step 2: Placebo test ──────────────────────────────────────────────────
    logger.info("Step 2: Placebo test for poverty gap outcomes")
    placebo_results = run_placebo_poverty_gap(panel)
    placebo_results.to_csv(OUTPUT_DIR / "placebo_poverty_gap.csv", index=False)
    logger.info("Placebo results saved")

    # Check if any outcome failed placebo
    failed_placebo = []
    passed_placebo = []
    for _, row in placebo_results.iterrows():
        if row["verdict"] == "WARNING":
            failed_placebo.append(row["outcome"])
            logger.warning(
                "%s FAILED placebo (WCB p=%.4f) — will be reported "
                "as appendix only",
                row["outcome"], row["pval_wbt"],
            )
        else:
            passed_placebo.append(row["outcome"])
            logger.info(
                "%s PASSED placebo (WCB p=%.4f) — included in main analysis",
                row["outcome"], row["pval_wbt"],
            )

    if not passed_placebo:
        logger.warning(
            "All poverty gap outcomes failed placebo. "
            "Reporting as appendix only — no causal DiD."
        )

    # ── Step 3: Event study ───────────────────────────────────────────────────
    logger.info("Step 3: Event study for poverty gap outcomes")
    run_event_study_poverty_gap(panel)

    # ── Step 4: Baseline DiD ──────────────────────────────────────────────────
    logger.info("Step 4: Baseline DiD — poverty gap outcomes")

    results_baseline = run_did_poverty_gap(
        panel, DID_POST_YEARS_BASELINE, "baseline_2021_2025"
    )
    results_covid = run_did_poverty_gap(
        panel, DID_POST_YEARS_COVID, "covid_robust_2022_2025"
    )

    combined = pd.concat(
        [results_baseline, results_covid], ignore_index=True
    )
    combined.to_csv(OUTPUT_DIR / "did_poverty_gap_all_specs.csv", index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PRIMARY SPEC SUMMARY ({PRIMARY_SPEC})")
    print(f"{'='*70}")
    primary_rows = combined[combined["exposure_spec"] == PRIMARY_SPEC]
    for _, row in primary_rows.iterrows():
        stars = (
            "***" if row["pval_wbt"] < 0.01
            else "**" if row["pval_wbt"] < 0.05
            else "*"  if row["pval_wbt"] < 0.10
            else ""
        ) if not pd.isna(row["pval_wbt"]) else "n/a"
        print(
            f"  {row['label']:30s} | {row['outcome']:15s} | "
            f"β={row['coef']:+.5f} | SE={row['se']:.5f} | "
            f"p_WCB={row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else
            f"  {row['label']:30s} | {row['outcome']:15s} | "
            f"β={row['coef']:+.5f} | SE={row['se']:.5f} | p_WCB=n/a"
        )

    logger.info(
        "Poverty gap analysis complete. Results saved to %s", OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
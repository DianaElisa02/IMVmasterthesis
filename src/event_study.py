"""
event_study.py
==============
Event study, placebo test, and region-specific time trend robustness
for the IMV DiD analysis. All estimation via PyFixest (feols + wildboottest),
following the same stack as binned_did.py.

Specifications
--------------
1. Event study (main):
   Y_hrt = sum_t!=2019 beta_t (yr_t x Exposure_r) + gamma_r + delta_t + X theta + e
   Reference year: 2019. Pre-trend test: joint Wald on yr_2017 and yr_2018 interactions.

2. Placebo test:
   Uses only pre-reform years (2017, 2018, 2019).
   Fake post = 1 if year == 2019, 0 if year in {2017, 2018}.
   Reference year (year FE): 2018.
   Tests H0: beta_placebo = 0 via WCB. If not rejected, parallel trends hold.

3. Region-specific time trends robustness:
   Adds drgn2 x year_centered interaction terms to the event study formula.
   Documented as underpowered with 17 clusters — run once for completeness,
   do not report as primary evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

from src.constants import (
    ANALYSIS_OUTCOMES,
    BALANCE_CONTROLS,
    EVENT_STUDY_REFERENCE_YEAR,
    EVENT_STUDY_YEARS,
    EXPOSURE_SPECS,
    PLACEBO_FAKE_TREATMENT_YEAR,
    PLACEBO_REFERENCE_YEAR,
    PLACEBO_YEARS,
    YEARS,
)

logger = logging.getLogger(__name__)

PRIMARY_SPEC  = EXPOSURE_SPECS[0]
_PRE_YEARS    = YEARS                        # [2017, 2018, 2019]
_REF_YEAR     = EVENT_STUDY_REFERENCE_YEAR  # 2019
_EVENT_YEARS  = EVENT_STUDY_YEARS           # [2017, 2018, 2021, 2022, 2023, 2024, 2025]


# ---------------------------------------------------------------------------
# FIX 1: exposure parameter is now explicit (not hard-coded to PRIMARY_SPEC)
# ---------------------------------------------------------------------------
def build_event_study_data(panel: pl.DataFrame, exposure: str = PRIMARY_SPEC) -> pd.DataFrame:
    """
    Filter panel to event-study years and build interaction columns.

    Parameters
    ----------
    panel : pl.DataFrame
        Full analysis panel in Polars format.
    exposure : str
        Column name of the exposure variable to use. Defaults to PRIMARY_SPEC.
        Pass any entry from EXPOSURE_SPECS to run robustness checks.
    """
    keep = _PRE_YEARS + _EVENT_YEARS
    df = panel.filter(pl.col("year").is_in(keep)).to_pandas()

    # Store exposure under a canonical name so run_event_study is agnostic
    df["_exposure"] = df[exposure]

    for yr in _EVENT_YEARS:
        df[f"yr_{yr}"] = (df["year"] == yr).astype(float)
        df[f"yr_{yr}_x_exp"] = df[f"yr_{yr}"] * df["_exposure"]

    logger.info(
        "Event study data: %d obs | years: %s | reference: %d | exposure: %s",
        len(df),
        sorted(df["year"].unique().tolist()),
        _REF_YEAR,
        exposure,
    )

    return df


# ---------------------------------------------------------------------------
# FIX 2: model_tag parameter makes baseline vs region-trends outputs distinct
# FIX 3: lead magnitude diagnostics added (lead_mean, lead_max_abs)
# FIX 4: collinearity severity warning surfaced explicitly
# ---------------------------------------------------------------------------
def run_event_study(
    df: pd.DataFrame,
    outcome: str,
    controls: list[str],
    output_dir: Path,
    region_trends: bool = False,
    model_tag: str = "baseline",
) -> pd.DataFrame:
    """
    Estimate the event study for one outcome via pyfixest feols + WCB.

    Formula:
        outcome ~ yr_2017_x_exp + yr_2018_x_exp + yr_2021_x_exp + ...
                  [+ controls] [+ region x year_centered]
                  | drgn2 + year

    Clusters: drgn2 (CRV1).
    WCB: wildboottest with Webb weights, B=9999, per interaction term.
    Pre-trend Wald test: joint H0 that yr_2017_x_exp = yr_2018_x_exp = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_event_study_data().
    outcome : str
        Outcome column name.
    controls : list[str]
        Additional control columns (e.g. BALANCE_CONTROLS).
    output_dir : Path
        Directory for CSV and plot outputs.
    region_trends : bool
        If True, add region-specific linear time trends (robustness only).
    model_tag : str
        Label embedded in output filenames and the 'model' column.
        Use "baseline" for the primary event study and "region_trends" for the
        robustness variant so results are never mixed.
    """
    # FIX 2: enforce consistent tag when region_trends=True
    if region_trends:
        model_tag = "region_trends"

    interaction_terms = [f"yr_{yr}_x_exp" for yr in _EVENT_YEARS]
    ctrl_str = (" + " + " + ".join(controls)) if controls else ""

    if region_trends:
        # Center year at reference year to reduce collinearity
        df = df.copy()
        df["year_c"] = df["year"] - _REF_YEAR
        region_list = sorted(df["drgn2"].unique().tolist())
        # Omit one region trend (smallest drgn2) as the reference
        ref_region = region_list[0]
        trend_terms = []
        for reg in region_list[1:]:
            col = f"trend_r{reg}"
            df[col] = ((df["drgn2"] == reg).astype(float)) * df["year_c"]
            trend_terms.append(col)
        trend_str = " + " + " + ".join(trend_terms)
        logger.info(
            "  Region trends: %d terms added (reference region: %d)",
            len(trend_terms), ref_region,
        )
    else:
        trend_str = ""

    formula = (
        f"{outcome} ~ "
        + " + ".join(interaction_terms)
        + ctrl_str
        + trend_str
        + " | drgn2 + year"
    )
    logger.info("  Formula [%s]: %s", model_tag, formula)

    trend_cols = [c for c in df.columns if c.startswith("trend_")] if region_trends else []
    keep_cols = [outcome, "drgn2", "year"] + interaction_terms + controls + trend_cols
    df_clean = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)

    X_cols = interaction_terms + controls + trend_cols
    X = df_clean[[c for c in X_cols if c in df_clean.columns]]
    rank = np.linalg.matrix_rank(X.values)

    # FIX 4: graduated collinearity warning
    if rank < X.shape[1]:
        severity = rank / X.shape[1]
        if region_trends and severity < 0.90:
            logger.warning(
                "  [%s] Region trends specification is highly collinear "
                "(rank=%d / %d cols = %.0f%%) — treat as robustness only.",
                model_tag, rank, X.shape[1], severity * 100,
            )
        else:
            logger.warning(
                "  [%s] Design matrix rank-deficient: rank=%d, cols=%d. "
                "Results may be unreliable.",
                model_tag, rank, X.shape[1],
            )

    fit = pf.feols(formula, data=df_clean, vcov={"CRV1": "drgn2"})
    logger.info(
        "  [%s] Estimated: %d obs, %d clusters",
        model_tag, len(df_clean), df_clean["drgn2"].nunique(),
    )

    # ------------------------------------------------------------------
    # Collect per-year coefficients + WCB p-values
    # ------------------------------------------------------------------
    rows = []
    for yr in _EVENT_YEARS:
        term = f"yr_{yr}_x_exp"
        coef = float(fit.coef()[term])
        se   = float(fit.se()[term])
        pval = float(fit.pvalue()[term])

        p_wbt = np.nan
        try:
            boot  = fit.wildboottest(param=term, reps=9999, seed=42 + yr)
            raw   = boot["Pr(>|t|)"]
            p_wbt = float(raw.iloc[0]) if hasattr(raw, "iloc") else float(raw)
        except Exception as e:
            logger.warning("    WCB failed for %s %d: %s", outcome, yr, e)

        rows.append({
            "year":       yr,
            "rel_year":   yr - _REF_YEAR,
            "coef":       coef,
            "se":         se,
            "pval_crv1":  pval,
            "pval_wbt":   p_wbt,
            "ci_low":     coef - 2.12 * se,
            "ci_high":    coef + 2.12 * se,
            "pre_period": yr < 2020,
        })

    # Reference year: coefficient normalised to zero
    rows.append({
        "year":       _REF_YEAR,
        "rel_year":   0,
        "coef":       0.0,
        "se":         0.0,
        "pval_crv1":  np.nan,
        "pval_wbt":   np.nan,
        "ci_low":     0.0,
        "ci_high":    0.0,
        "pre_period": True,
    })

    coef_df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Pre-trend Wald test (joint H0: all lead coefficients = 0)
    # ------------------------------------------------------------------
    pre_terms   = [f"yr_{yr}_x_exp" for yr in _EVENT_YEARS if yr < 2020]
    coef_names  = fit.coef().index.tolist()
    pre_indices = [coef_names.index(t) for t in pre_terms if t in coef_names]

    if len(pre_indices) >= 1:
        R = np.zeros((len(pre_indices), len(coef_names)))
        for i, j in enumerate(pre_indices):
            R[i, j] = 1.0
        try:
            wald      = fit.wald_test(R=R)
            wald_stat = float(wald["statistic"])
            wald_p    = float(wald["pvalue"])
            logger.info(
                "  [%s] Pre-trend Wald [%s]: stat=%.3f p=%.4f %s",
                model_tag, outcome, wald_stat, wald_p,
                "✓ pre-trends not rejected" if wald_p > 0.10 else "⚠ pre-trends rejected",
            )
            if region_trends and wald_p < 0.10:
                logger.warning(
                    "  [%s] Region trends specification shows pre-trend rejection — "
                    "interpret as robustness only (low power, 17 clusters).",
                    model_tag,
                )
        except Exception as e:
            logger.warning("  Pre-trend Wald test failed: %s", e)
            wald_stat, wald_p = np.nan, np.nan
    else:
        wald_stat, wald_p = np.nan, np.nan

    # FIX 3: lead magnitude diagnostics
    lead_coefs = [
        float(fit.coef()[t]) for t in pre_terms if t in fit.coef().index
    ]
    lead_mean    = float(np.mean(lead_coefs))    if lead_coefs else np.nan
    lead_max_abs = float(np.max(np.abs(lead_coefs))) if lead_coefs else np.nan

    logger.info(
        "  [%s] Lead diagnostics [%s]: mean=%.4f, max_abs=%.4f",
        model_tag, outcome, lead_mean if not np.isnan(lead_mean) else float("nan"),
        lead_max_abs if not np.isnan(lead_max_abs) else float("nan"),
    )

    # FIX 2: tag all rows with the model label and diagnostics
    coef_df["model"]              = model_tag
    coef_df["pretrend_wald_stat"] = wald_stat
    coef_df["pretrend_wald_p"]    = wald_p
    coef_df["lead_mean"]          = lead_mean
    coef_df["lead_max_abs"]       = lead_max_abs

    # ------------------------------------------------------------------
    # Save coefficient table
    # ------------------------------------------------------------------
    tag      = model_tag   # "baseline" or "region_trends"
    out_path = output_dir / f"{tag}_{outcome}.csv"
    coef_df.to_csv(out_path, index=False)
    logger.info("  Saved: %s", out_path)

    # FIX 3: save compact pre-trend summary for thesis appendix table
    pretrend_summary = pd.DataFrame([{
        "outcome":      outcome,
        "model":        model_tag,
        "wald_stat":    wald_stat,
        "wald_p":       wald_p,
        "lead_mean":    lead_mean,
        "lead_max_abs": lead_max_abs,
        "n_pre_terms":  len(pre_indices),
        "n_obs":        len(df_clean),
        "n_clusters":   df_clean["drgn2"].nunique(),
    }])
    summary_path = output_dir / f"pretrend_summary_{tag}_{outcome}.csv"
    pretrend_summary.to_csv(summary_path, index=False)
    logger.info("  Pre-trend summary saved: %s", summary_path)

    return coef_df


def run_placebo(
    panel: pl.DataFrame,
    outcome: str,
    controls: list[str],
    output_dir: Path,
) -> dict:
    """
    Placebo test using only pre-reform years (PLACEBO_YEARS).

    Fake post = 1 if year == PLACEBO_FAKE_TREATMENT_YEAR, 0 otherwise.
    Reference year for year FE: PLACEBO_REFERENCE_YEAR.
    Tests H0: beta_placebo = 0 via WCB.
    """
    df = (
        panel
        .filter(pl.col("year").is_in(PLACEBO_YEARS))
        .to_pandas()
    )

    df["post_fake"]       = (df["year"] == PLACEBO_FAKE_TREATMENT_YEAR).astype(float)
    df["post_fake_x_exp"] = df["post_fake"] * df[PRIMARY_SPEC]

    ctrl_str = (" + " + " + ".join(controls)) if controls else ""
    formula  = f"{outcome} ~ post_fake_x_exp{ctrl_str} | drgn2 + year"

    df_clean = df[
        [c for c in [outcome, "drgn2", "year", "post_fake_x_exp"] + controls
         if c in df.columns]
    ].dropna().reset_index(drop=True)

    fit  = pf.feols(formula, data=df_clean, vcov={"CRV1": "drgn2"})
    coef = float(fit.coef()["post_fake_x_exp"])
    se   = float(fit.se()["post_fake_x_exp"])
    pval = float(fit.pvalue()["post_fake_x_exp"])

    p_wbt = np.nan
    try:
        boot  = fit.wildboottest(param="post_fake_x_exp", reps=9999, seed=42)
        raw   = boot["Pr(>|t|)"]
        p_wbt = float(raw.iloc[0]) if hasattr(raw, "iloc") else float(raw)
    except Exception as e:
        logger.warning("  Placebo WCB failed (%s): %s", outcome, e)

    result = {
        "outcome":    outcome,
        "coef":       coef,
        "se":         se,
        "pval_crv1":  pval,
        "pval_wbt":   p_wbt,
        "n_obs":      len(df_clean),
        "n_clusters": df_clean["drgn2"].nunique(),
    }
    logger.info(
        "  Placebo [%s]: coef=%+.4f SE=%.4f p_CRV1=%.4f p_WCB=%.4f %s",
        outcome, coef, se, pval, p_wbt,
        "✓ placebo null not rejected" if p_wbt > 0.10 else "placebo significant",
    )

    out_path = output_dir / f"placebo_{outcome}.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    return result


def plot_event_study(
    coef_df: pd.DataFrame,
    outcome: str,
    output_dir: Path,
    title_suffix: str = "",
) -> None:
    """
    Standard event study coefficient plot with 95% CI.
    Pre-reform years in blue, post-reform in orange, reference year in teal.
    """
    df = coef_df.sort_values("year").copy()

    fig, ax = plt.subplots(figsize=(11, 5))

    pre  = df[df["rel_year"] <  0]
    ref  = df[df["rel_year"] == 0]
    post = df[df["rel_year"] >  0]

    for subset, color, label in [
        (pre,  "#378ADD", "Pre-reform"),
        (post, "#F4A261", "Post-reform"),
    ]:
        ax.errorbar(
            subset["rel_year"], subset["coef"],
            yerr=subset["coef"] - subset["ci_low"],
            fmt="o", color=color, capsize=4, linewidth=1.5,
            markersize=6, label=label,
        )

    ax.scatter(ref["rel_year"], ref["coef"], color="#2A9D8F",
               zorder=5, s=80, label=f"Reference ({_REF_YEAR})")

    ax.axhline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="#B4B2A9", linewidth=0.8, linestyle=":",
               label="Reform (mid-2020)")

    outcome_label = {
        "vhMATDEP":      "Severe material deprivation",
        "vhPobreza":     "At-risk-of-poverty",
        "poverty_gap":   "Poverty gap (FGT-1)",
        "poverty_gap_sq":"Poverty gap squared (FGT-2)",
    }.get(outcome, outcome)

    wald_p    = df["pretrend_wald_p"].dropna().iloc[0] if "pretrend_wald_p" in df.columns else None
    lead_mean = df["lead_mean"].dropna().iloc[0]       if "lead_mean" in df.columns else None
    model     = df["model"].iloc[0]                    if "model" in df.columns else ""

    wald_note = (
        f"Wald p={wald_p:.3f}  |  lead mean={lead_mean:+.4f}"
        if wald_p is not None and lead_mean is not None
        else ""
    )

    ax.set_xlabel("Years relative to reform (reference: 2019)", fontsize=11)
    ax.set_ylabel("Coefficient (pp change per SD exposure)", fontsize=11)
    ax.set_title(
        f"Event study — {outcome_label}{title_suffix}\n"
        f"Model: {model}  |  {wald_note}",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    tag      = model if model else ("region_trends" if "region_trends" in title_suffix.lower() else "event_study")
    out_path = output_dir / f"{tag}_{outcome}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Plot saved: %s", out_path)
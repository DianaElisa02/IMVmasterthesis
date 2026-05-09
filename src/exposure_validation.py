"""
exposure_validation.py
======================
Statistical validation of the IMV counterfactual simulation.
 
  1. Benefit amount bounds vs statutory GMI amounts
  2. Monotonicity: larger households receive more
  3. Income means test: recipients have lower income than non-recipients
  4. Cross-year distributional consistency (Kolmogorov-Smirnov)
  5. Regional rank consistency (Spearman) across waves
  6. Benefit formula plausibility check vs statutory single-person GMI
"""
 
from __future__ import annotations
 
import logging
from pathlib import Path
 
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
 
logger = logging.getLogger(__name__)
 
 
def test_benefit_bounds(
    df: pd.DataFrame,
    year: int,
    statutory_min: float,
    statutory_max: float,
) -> dict:
    recipients = df[df["bsa00_s"] > 0].copy()
    n = len(recipients)
    w = recipients["dwt"].sum()
 
    below_floor = (recipients["bsa00_s"] < 10).sum()
    above_max   = (recipients["bsa00_s"] > statutory_max * 1.10).sum()  # 10% tolerance
    wmean       = (recipients["bsa00_s"] * recipients["dwt"]).sum() / w
 
    result = {
        "test": "benefit_bounds",
        "year": year,
        "n_recipients_unweighted": n,
        "n_recipients_weighted": round(w, 0),
        "mean_monthly_benefit": round(wmean, 2),
        "statutory_single_person_GMI": statutory_min,
        "statutory_max_GMI": statutory_max,
        "n_below_floor": int(below_floor),
        "n_above_max_110pct": int(above_max),
        "pass": below_floor == 0 and above_max == 0,
    }
    status = "PASS" if result["pass"] else "WARN"
    logger.info(
        "[%s] Test 1 — Benefit bounds %d: mean=€%.2f, below_floor=%d, above_max=%d",
        status, year, wmean, below_floor, above_max,
    )
    return result
 
def test_monotonicity(df: pd.DataFrame, year: int) -> dict:

    recipients = df[df["bsa00_s"] > 0].copy()

    hh_size = df.groupby("idhh")["idperson"].count().rename("hh_size_proxy")
    recipients = recipients.merge(hh_size, on="idhh", how="left")

    rho, pval = spearmanr(
        recipients["hh_size_proxy"],
        recipients["bsa00_s"]
    )

    by_size = (
        recipients.groupby("hh_size_proxy")
        .apply(lambda x: round(
            (x["bsa00_s"] * x["dwt"]).sum() / x["dwt"].sum(), 2
        ))
        .reset_index()
        .rename(columns={0: "mean_bsa00_s"})
    )

    result = {
        "test": "monotonicity",
        "year": year,
        "spearman_rho_hsize_benefit": round(rho, 3),
        "spearman_p": round(pval, 4),
        "mean_by_hsize": by_size.to_dict("records"),
        "pass": rho > 0 and pval < 0.05,
    }
    status = "PASS" if result["pass"] else "WARN"
    logger.info(
        "[%s] Test 2 — Monotonicity %d: rho=%.3f (p=%.4f)",
        status, year, rho, pval,
    )
    return result
 
 
def test_income_means_test(df: pd.DataFrame, year: int) -> dict:
    recipients     = df[df["bsa00_s"] > 0]["yds"].dropna()
    non_recipients = df[df["bsa00_s"] == 0]["yds"].dropna()
 
    stat, pval = mannwhitneyu(
        recipients, non_recipients, alternative="less"
    )
 
    mean_inc_rec    = recipients.mean()
    mean_inc_nonrec = non_recipients.mean()
 
    result = {
        "test": "income_means_test",
        "year": year,
        "mean_yds_recipients": round(mean_inc_rec, 2),
        "mean_yds_non_recipients": round(mean_inc_nonrec, 2),
        "ratio": round(mean_inc_rec / mean_inc_nonrec, 3),
        "mannwhitney_stat": round(stat, 0),
        "mannwhitney_p": round(pval, 6),
        "pass": pval < 0.05 and mean_inc_rec < mean_inc_nonrec,
    }
    status = "PASS" if result["pass"] else "WARN"
    logger.info(
        "[%s] Test 3 — Income means test %d: "
        "mean_yds_rec=€%.0f vs non_rec=€%.0f (p=%.4f)",
        status, year,
        mean_inc_rec, mean_inc_nonrec, pval,
    )
    return result
 
 
def test_cross_year_consistency(
    imv_dfs: dict[int, pd.DataFrame],
) -> list[dict]:
    results = []
    years = sorted(imv_dfs.keys())
 
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        dist1 = imv_dfs[y1][imv_dfs[y1]["bsa00_s"] > 0]["bsa00_s"].dropna()
        dist2 = imv_dfs[y2][imv_dfs[y2]["bsa00_s"] > 0]["bsa00_s"].dropna()
 
        stat, pval = ks_2samp(dist1, dist2)
 
        result = {
            "test": "cross_year_ks",
            "years": f"{y1}_vs_{y2}",
            "ks_statistic": round(stat, 4),
            "ks_p_value": round(pval, 4),
            "mean_y1": round(dist1.mean(), 2),
            "mean_y2": round(dist2.mean(), 2),
            # high p-value means distributions are similar — desirable
            "pass": pval > 0.05,
            "note": "high p-value = distributions similar across years",
        }
        status = "PASS" if result["pass"] else "INFO"
        logger.info(
            "[%s] Test 4 — KS consistency %d vs %d: stat=%.4f (p=%.4f)",
            status, y1, y2, stat, pval,
        )
        results.append(result)
 
    return results
 
 
def test_regional_rank_consistency(
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
) -> list[dict]:
    def regional_means(df: pd.DataFrame) -> pd.Series:
        rec = df[
            (df["bsa00_s"] > 0) &
            (~df["drgn2"].isin(exclude_regions))
        ].copy()
        return (
            rec.groupby("drgn2")
            .apply(lambda x:
                   (x["bsa00_s"] * x["dwt"]).sum() / x["dwt"].sum()
            )
        )
 
    years = sorted(imv_dfs.keys())
    results = []
 
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        m1 = regional_means(imv_dfs[y1])
        m2 = regional_means(imv_dfs[y2])
 
        common = m1.index.intersection(m2.index)
        rho, pval = spearmanr(m1[common], m2[common])
 
        result = {
            "test": "regional_rank_consistency",
            "years": f"{y1}_vs_{y2}",
            "spearman_rho": round(rho, 3),
            "spearman_p": round(pval, 4),
            "n_regions": len(common),
            "pass": rho > 0.7 and pval < 0.05,
        }
        status = "PASS" if result["pass"] else "WARN"
        logger.info(
            "[%s] Test 5 — Regional rank %d vs %d: rho=%.3f (p=%.4f)",
            status, y1, y2, rho, pval,
        )
        results.append(result)
 
    return results
 
def test_formula_plausibility(
    df: pd.DataFrame,
    year: int,
    statutory_single: float,
) -> dict:
    """
    Test 6: Benefit formula plausibility.
    Single-person household recipients should receive approximately
    the statutory single-person GMI amount (within 20% tolerance).
    Uses person count per idhh as household size proxy.
    """
    # Compute household size proxy
    hh_size = df.groupby("idhh")["idperson"].count().rename("hh_size_proxy")
    df_merged = df.merge(hh_size, on="idhh", how="left")

    single_rec = df_merged[
        (df_merged["bsa00_s"] > 0) &
        (df_merged["hh_size_proxy"] == 1)
    ].copy()

    if len(single_rec) == 0:
        return {
            "test": "formula_plausibility",
            "year": year,
            "pass": None,
            "note": "no single-person recipients found",
        }

    wmean = (
        (single_rec["bsa00_s"] * single_rec["dwt"]).sum() /
        single_rec["dwt"].sum()
    )
    pct_diff = abs(wmean - statutory_single) / statutory_single

    result = {
        "test": "formula_plausibility",
        "year": year,
        "mean_bsa00_s_single_hh": round(wmean, 2),
        "statutory_single_GMI": statutory_single,
        "pct_difference": round(100 * pct_diff, 1),
        "n_single_recipients": len(single_rec),
        "pass": pct_diff <= 0.20,
        "note": "within 20% of statutory single-person GMI",
    }
    status = "PASS" if result["pass"] else "WARN"
    logger.info(
        "[%s] Test 6 — Formula plausibility %d: "
        "mean_single=€%.2f vs statutory=€%.2f (diff=%.1f%%)",
        status, year, wmean, statutory_single, 100 * pct_diff,
    )
    return result

def run_validation(
    imv_dfs: dict[int, pd.DataFrame],
    statutory_single: float,
    statutory_max: float,
    exclude_regions: frozenset[int],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []
 
    for year, df in sorted(imv_dfs.items()):
        logger.info("=" * 50)
        logger.info("Validating IMV simulation — year %d", year)
 
        all_results.append(
            test_benefit_bounds(df, year, statutory_single, statutory_max)
        )
        all_results.append(
            test_monotonicity(df, year)
        )
        all_results.append(
            test_income_means_test(df, year)
        )
        all_results.append(
            test_formula_plausibility(df, year, statutory_single)
        )
 
    all_results.extend(test_cross_year_consistency(imv_dfs))
    all_results.extend(
        test_regional_rank_consistency(imv_dfs, exclude_regions)
    )
 
    flat_results = []
    for r in all_results:
        flat = {k: v for k, v in r.items()
                if not isinstance(v, (list, dict))}
        flat_results.append(flat)
 
    results_df = pd.DataFrame(flat_results)
 
    out_path = output_dir / "imv_validation_report.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Validation report saved → %s", out_path)
 
    passed = results_df["pass"].sum()
    total  = results_df["pass"].notna().sum()
    logger.info("Validation summary: %d/%d tests passed", passed, total)
 
    return results_df
 
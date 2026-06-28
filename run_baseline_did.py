"""
run_baseline_did.py

Runner for the complementary continuous-intensity DiD estimation.

Runs two post-period windows separately:
  1. Full post-reform period: 2021–2025
  2. COVID-robust period: 2022–2025

The tercile-based DiD remains the baseline specification. This runner reports
continuous-intensity estimates as complementary dose-response evidence.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.baseline_did import build_did_data, run_baseline_did
from src.constants import (
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "continuous_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPECS = [
    "exposure_cov_hybrid",
    "exposure_exp_hybrid",
]


def _stars(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def print_results(df: pd.DataFrame, label: str) -> None:
    """Print a compact results table to stdout."""
    sep = "=" * 86
    print(f"\n{sep}")
    print(f"  Complementary continuous-intensity DiD — {label}")
    print("  Inference: WCB p-value (primary) | CRV1 p-value (auxiliary)")
    print("  Confidence intervals use t(G−1), with G taken from each fitted model")
    print(sep)
    print(
        f"  {'Outcome':<12} {'Exposure spec':<32} "
        f"{'β':>8} {'SE':>7} {'CI':>22} "
        f"{'G':>4} {'p_CRV1':>8} {'p_WCB':>8}"
    )
    print("-" * 86)

    for _, row in df.iterrows():
        stars = _stars(row["pval_wbt"])
        ci_str = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
        p_wbt_str = (
            f"{row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else "unavailable"
        )
        print(
            f"  {row['outcome']:<12} {row['exposure_spec']:<32} "
            f"{row['coef']:+8.4f} {row['se']:7.4f} {ci_str:>22} "
            f"{int(row['n_clusters']):4d} {row['pval_cluster']:8.4f} {p_wbt_str:>14}"
        )


def print_primary_summary(df: pd.DataFrame) -> None:
    """Print summaries for both co-primary hybrid exposure margins."""
    sep = "=" * 86
    print(f"\n{sep}")
    print("  COMPLEMENTARY CONTINUOUS SUMMARY — HYBRID COVERAGE AND BENEFIT MARGINS")
    print(sep)

    primary = df[df["exposure_spec"].isin(PRIMARY_SPECS)]
    for _, row in primary.iterrows():
        stars = _stars(row["pval_wbt"])
        p_wbt_str = (
            f"{row['pval_wbt']:.4f} {stars}"
            if not pd.isna(row["pval_wbt"])
            else "n/a"
        )
        print(
            f"  {row['label']:<30} | {row['exposure_spec']:<25} | "
            f"{row['outcome']:<12} | β={row['coef']:+.4f}  "
            f"SE={row['se']:.4f}  p_WCB={p_wbt_str}"
        )


def main() -> None:
    logger.info("=== IMV DiD — complementary continuous-intensity models ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d observations", len(panel))

    logger.info("--- Continuous DiD: post = 2021–2025 ---")
    did_full = build_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    results_full = run_baseline_did(did_full, label="continuous_2021_2025")

    print_results(results_full, "Full post-reform period (2021–2025)")
    full_path = OUTPUT_DIR / "continuous_did_2021_2025.csv"
    results_full.to_csv(full_path, index=False)
    logger.info("Saved: %s", full_path)

    logger.info("--- Continuous COVID robustness: post = 2022–2025 ---")
    did_covid = build_did_data(panel, post_years=DID_POST_YEARS_COVID)
    results_covid = run_baseline_did(
        did_covid,
        label="continuous_covid_robust_2022_2025",
    )

    print_results(results_covid, "COVID robust — post = 2022–2025")
    covid_path = OUTPUT_DIR / "continuous_did_2022_2025.csv"
    results_covid.to_csv(covid_path, index=False)
    logger.info("Saved: %s", covid_path)

    combined = pd.concat([results_full, results_covid], ignore_index=True)
    combined_path = OUTPUT_DIR / "continuous_did_all_specs.csv"
    combined.to_csv(combined_path, index=False)
    logger.info("Saved combined: %s", combined_path)

    print_primary_summary(combined)

    logger.info("Continuous DiD complete. All results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

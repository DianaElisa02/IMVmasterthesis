"""
run_binned_did.py
=================
Runner for the binned DiD specification.

Runs two post-period windows:
  1. Baseline     : post = 2021-2025
  2. COVID robust : post = 2022-2025

For each window, estimates all EXPOSURE_SPECS x ANALYSIS_OUTCOMES (matdep, poverty).
Tercile assignment is recomputed dynamically for each exposure spec.

Reads from analysis_dataset.parquet (primary analysis dataset).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import run_binned_did
from src.constants import (
    ANALYSIS_OUTCOMES,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_SPECS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "binned_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_SPEC = EXPOSURE_SPECS[0]

def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def print_results(df: pd.DataFrame, label: str) -> None:
    if df.empty or "outcome" not in df.columns:
        logger.warning("No results to print for %s", label)
        return

    sep  = "=" * 80
    sep2 = "-" * 65

    print(f"\n{sep}")
    print(f"  Binned DiD — {label}")
    print(f"  Inference: WCB p-value (primary) | CRV1 p-value (auxiliary)")
    print(sep)

    for spec in EXPOSURE_SPECS:
        spec_rows = df[df["exposure_spec"] == spec]
        if spec_rows.empty:
            continue

        print(f"\n  Exposure spec: {spec}")
        print(f"  {sep2}")

        for _, row in spec_rows.iterrows():
            stars_M = _stars(row["pval_wbt_medium"])
            stars_H = _stars(row["pval_wbt_high"])

            print(f"\n    Outcome: {row['outcome']}")

            # Medium tercile
            if not pd.isna(row["pval_wbt_medium"]):
                print(
                    f"    b_M (medium vs low): {row['coef_medium']:+.4f}  "
                    f"SE={row['se_medium']:.4f}  "
                    f"CI=[{row['ci_low_medium']:+.4f}, {row['ci_high_medium']:+.4f}]  "
                    f"p_CRV1={row['pval_cluster_medium']:.4f}  "
                    f"p_WCB={row['pval_wbt_medium']:.4f} {stars_M}"
                )
            else:
                print(
                    f"    b_M (medium vs low): {row['coef_medium']:+.4f}  "
                    f"SE={row['se_medium']:.4f}  WCB unavailable"
                )

            # High tercile
            if not pd.isna(row["pval_wbt_high"]):
                print(
                    f"    b_H (high vs low)  : {row['coef_high']:+.4f}  "
                    f"SE={row['se_high']:.4f}  "
                    f"CI=[{row['ci_low_high']:+.4f}, {row['ci_high_high']:+.4f}]  "
                    f"p_CRV1={row['pval_cluster_high']:.4f}  "
                    f"p_WCB={row['pval_wbt_high']:.4f} {stars_H}"
                )
            else:
                print(
                    f"    b_H (high vs low)  : {row['coef_high']:+.4f}  "
                    f"SE={row['se_high']:.4f}  WCB unavailable"
                )

            # Linearity diagnostic
            ratio_str = (
                f"{row['linearity_ratio']:+.2f}"
                if not pd.isna(row["linearity_ratio"]) else "n/a"
            )
            if not pd.isna(row["linearity_stat"]):
                lin_verdict = "linear" if row["linearity_p"] > 0.10 else "NONLINEAR"
                print(
                    f"    Linearity b_H/b_M = {ratio_str}  |  "
                    f"Wald stat={row['linearity_stat']:.2f}  "
                    f"p={row['linearity_p']:.4f}  [{lin_verdict}]"
                )
            else:
                print(f"    Linearity b_H/b_M = {ratio_str}  |  Wald test unavailable")


def print_primary_summary(combined: pd.DataFrame) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  PRIMARY SPEC SUMMARY — {PRIMARY_SPEC}")
    print(f"  b_H (high vs low) | outcome x window")
    print(sep)

    primary = combined[combined["exposure_spec"] == PRIMARY_SPEC]
    for _, row in primary.iterrows():
        stars = _stars(row["pval_wbt_high"])
        print(
            f"  {row['label']:<30} | {row['outcome']:<10} | "
            f"b_H={row['coef_high']:+.4f}  SE={row['se_high']:.4f}  "
            f"p_WCB={row['pval_wbt_high']:.4f} {stars}"
        )


def main() -> None:
    logger.info("=== IMV DiD — run_binned_did.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))
    logger.info("Outcomes: %s", ANALYSIS_OUTCOMES)
    logger.info("Exposure specs: %s", EXPOSURE_SPECS)

    all_results = []

    logger.info("======= Baseline: post = 2021-2025 =======")
    results_baseline = run_binned_did(
        panel,
        post_years=DID_POST_YEARS_BASELINE,
        label="baseline_2021_2025",
        outcomes=ANALYSIS_OUTCOMES,
    )
    print_results(results_baseline, "Full post-reform period (2021-2025)")
    all_results.append(results_baseline)

    logger.info("======= COVID robust: post = 2022-2025 =======")
    results_covid = run_binned_did(
        panel,
        post_years=DID_POST_YEARS_COVID,
        label="covid_robust_2022_2025",
        outcomes=ANALYSIS_OUTCOMES,
    )
    print_results(results_covid, "COVID robust — post = 2022-2025")
    all_results.append(results_covid)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "binned_did_results.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "binned_did_results.csv")

    print_primary_summary(combined)

    logger.info("Binned DiD complete. Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
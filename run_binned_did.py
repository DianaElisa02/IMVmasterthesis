"""
py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import build_binned_did_data, run_binned_did
from src.constants import (
    ANALYSIS_OUTCOMES,
    DID_POST_YEARS_BASELINE,
    DID_POST_YEARS_COVID,
    EXPOSURE_TERCILES,
    REGION_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "binned_did"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_results(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*78}")
    print(f"  Binned DiD — {label}")
    print(f"  Reference: low-exposure tercile "
          f"(País Vasco, Navarra, Asturias, Cantabria, Illes Balears)")
    print(f"{'='*78}")

    for _, row in df.iterrows():
        print(f"\n  Outcome: {row['outcome']}")
        print(f"  {'-'*60}")

        # Medium tercile
        stars_M = (
            "***" if row["pval_wbt_medium"] < 0.01
            else "**" if row["pval_wbt_medium"] < 0.05
            else "*"  if row["pval_wbt_medium"] < 0.10
            else ""
        ) if not pd.isna(row["pval_wbt_medium"]) else "n/a"

        print(
            f"  β_M (medium vs low) : {row['coef_medium']:+.4f}  "
            f"SE={row['se_medium']:.4f}  "
            f"CI=[{row['ci_low_medium']:+.4f}, {row['ci_high_medium']:+.4f}]  "
            f"p_WCB={row['pval_wbt_medium']:.4f} {stars_M}"
        )

        # High tercile
        stars_H = (
            "***" if row["pval_wbt_high"] < 0.01
            else "**" if row["pval_wbt_high"] < 0.05
            else "*"  if row["pval_wbt_high"] < 0.10
            else ""
        ) if not pd.isna(row["pval_wbt_high"]) else "n/a"

        print(
            f"  β_H (high vs low)   : {row['coef_high']:+.4f}  "
            f"SE={row['se_high']:.4f}  "
            f"CI=[{row['ci_low_high']:+.4f}, {row['ci_high_high']:+.4f}]  "
            f"p_WCB={row['pval_wbt_high']:.4f} {stars_H}"
        )

        # Linearity diagnostic
        print(
            f"  Linearity ratio β_H/β_M : {row['linearity_ratio']:+.2f}  "
            f"(linear if ≈ 2.0)"
        )
        print(
            f"  Linearity test (β_H = 2β_M): "
            f"F={row['linearity_f']:.2f}, p={row['linearity_p']:.4f}  "
            f"{'✓ linear' if row['linearity_p'] > 0.10 else '⚠ NONLINEAR'}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — run_binned_did.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    # Log tercile composition
    logger.info("Tercile composition (FIXED in constants.py):")
    for label, regions in EXPOSURE_TERCILES.items():
        names = [REGION_NAMES.get(r, str(r)) for r in regions]
        logger.info("  %s: %s", label, names)

    all_results = []

    # ── Spec 1: Baseline (2021-2025) ──────────────────────────────────────────
    logger.info("--- Baseline binned DiD: 2021-2025 ---")
    did_baseline = build_binned_did_data(panel, post_years=DID_POST_YEARS_BASELINE)
    rows_b = []
    for outcome in ANALYSIS_OUTCOMES:
        result_dict, _, _ = run_binned_did(did_baseline, outcome=outcome)
        result_dict["label"] = "baseline_2021_2025"
        rows_b.append(result_dict)
    results_baseline = pd.DataFrame(rows_b)
    print_results(results_baseline, "Full post-reform period (2021-2025)")
    all_results.append(results_baseline)

    # ── Spec 2: COVID robust (2022-2025) ──────────────────────────────────────
    logger.info("--- COVID robust binned DiD: 2022-2025 ---")
    did_covid = build_binned_did_data(panel, post_years=DID_POST_YEARS_COVID)
    rows_c = []
    for outcome in ANALYSIS_OUTCOMES:
        result_dict, _, _ = run_binned_did(did_covid, outcome=outcome)
        result_dict["label"] = "covid_robust_2022_2025"
        rows_c.append(result_dict)
    results_covid = pd.DataFrame(rows_c)
    print_results(results_covid, "COVID robust — post-reform 2022-2025")
    all_results.append(results_covid)

    # ── Save combined ─────────────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "binned_did_results.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "binned_did_results.csv")

    # ── Interpretive summary ──────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  INTERPRETATION GUIDE")
    print(f"{'='*78}")
    print("  Linearity test passes (p > 0.10): continuous TWFE coefficient is")
    print("    a valid average — the linear dose-response assumption holds.")
    print("  Linearity test fails (p < 0.10): effects are nonlinear across the")
    print("    exposure distribution — interpret β_M and β_H separately.")
    print("")
    print("  If β_H is large and significant but β_M is null:")
    print("    → effect concentrated at top of distribution")
    print("    → continuous TWFE dilutes the high-exposure effect")
    print("")
    print("  If both β_M and β_H are null:")
    print("    → null result is genuine across the entire distribution")
    print("    → confirms the TWFE null is not an artefact of linearity")

    logger.info("Binned DiD complete. Results: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
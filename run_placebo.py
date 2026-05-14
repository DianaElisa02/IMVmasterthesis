"""
run_placebo.py
==============
Orchestrator for the IMV DiD placebo reform test.

Uses only pre-reform years (2017-2019). Assigns fake treatment at 2019.
Tests whether the exposure index picks up spurious pre-existing trends.

Usage:
  python run_placebo.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pandas as pd

from src.placebo import build_placebo_data, run_placebo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH  = Path("/workspaces/IMVmasterthesis")
INPUT_PATH = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR = BASE_PATH / "output" / "placebo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — run_placebo.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Full panel loaded: %d obs", len(panel))

    placebo = build_placebo_data(panel)
    logger.info("Placebo sample: %d obs", len(placebo))

    all_results = []

    for outcome in ["matdep", "poverty", "income_net_annual"]:
        logger.info("--- Placebo test: %s ---", outcome)

        result_table, result, wbt_results = run_placebo(
            placebo, outcome=outcome
        )

        all_results.append(result_table)

        # Print
        print(f"\n=== Placebo test — {outcome} ===")
        print(f"  Fake treatment year : 2019 (Post_fake=1)")
        print(f"  Reference year      : 2018 (omitted)")
        print(f"  Pre-reform years    : 2017, 2018, 2019")
        print(f"\n  Placebo coefficient : {result_table['coef'].iloc[0]:.4f}")
        print(f"  SE                  : {result_table['se'].iloc[0]:.4f}")
        print(f"  CI (t-dist)         : [{result_table['ci_low'].iloc[0]:.4f}, "
              f"{result_table['ci_high'].iloc[0]:.4f}]")
        print(f"  p-value (cluster)   : {result_table['pval_cluster'].iloc[0]:.4f}")
        print(f"  p-value (WCB)       : {result_table['pval_wbt'].iloc[0]:.4f}"
              if not pd.isna(result_table['pval_wbt'].iloc[0])
              else "  p-value (WCB)       : unavailable")
        print(f"\n  → {result_table['interpretation'].iloc[0]}")

    # Combined table
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "placebo_results.csv", index=False)
    logger.info("Saved: %s", OUTPUT_DIR / "placebo_results.csv")

    print("\n=== Placebo summary ===")
    print(combined[[
        "outcome", "coef", "se", "pval_cluster", "pval_wbt", "interpretation"
    ]].to_string(index=False))

    logger.info("Placebo test complete.")


if __name__ == "__main__":
    main()
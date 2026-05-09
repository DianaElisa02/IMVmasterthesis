"""
compute_exposure_variable.py
===================
Main entry point for the exposure variable pipeline.

Steps:
  1. Load all six EUROMOD output files (3 RMI + 3 IMV)
  2. Validate the IMV counterfactual simulation statistically
  3. Compute household-level gains pooled across 2017–2019
  4. Aggregate to regional exposure index
  5. Save outputs and produce chart

Usage
-----
    python compute_exposure.py

Outputs
-------
    output/exposure/imv_validation_report.csv
    output/exposure/exposure_index.csv
    output/exposure/household_gains.csv
    output/exposure/exposure_index.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("exposure.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

from src.constants import (
    RMI_FILES,
    IMV_FILES,
    EXPOSURE_EXCLUDE_REGIONS,
    EXPOSURE_OUTPUT_DIR,
    IMV_STATUTORY_2022,
    IMV_ADMIN_2022,
    REGION_NAMES,
    RMI_INCOMPATIBLE_REGIONS,
)
from src.exposure_loader import load_all_files
from src.exposure_validation import run_validation
from src.exposure_compute import (
    pool_gains,
    compute_exposure_index,
    plot_exposure,
    save_exposure,
)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting exposure variable pipeline")
    logger.info("=" * 60)

    logger.info("Step 1: Loading EUROMOD output files")
    rmi_dfs, imv_dfs = load_all_files(RMI_FILES, IMV_FILES)

    logger.info("Step 2: Validating IMV counterfactual simulation")
    validation_report = run_validation(
        imv_dfs=imv_dfs,
        statutory_single=IMV_STATUTORY_2022["basic_monthly"],
        statutory_max=IMV_STATUTORY_2022["max_monthly"],
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        output_dir=EXPOSURE_OUTPUT_DIR,
    )

    passed = validation_report["pass"].sum()
    total  = validation_report["pass"].notna().sum()
    logger.info(
        "Validation complete: %d/%d tests passed", passed, total
    )
    if passed < total:
        failed = validation_report[
            validation_report["pass"] == False
        ][["test", "year"]].to_string(index=False)
        logger.warning("Failed tests:\n%s", failed)

    logger.info("Step 3: Computing household-level gains")
    pooled = pool_gains(
        rmi_dfs=rmi_dfs,
        imv_dfs=imv_dfs,
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        incompatible_regions=RMI_INCOMPATIBLE_REGIONS,
    )

    # Quick gain distribution summary
    logger.info(
        "Pooled gain summary: mean=€%.2f | median=€%.2f | "
        "p25=€%.2f | p75=€%.2f | min=€%.2f | max=€%.2f",
        pooled["gain"].mean(),
        pooled["gain"].median(),
        pooled["gain"].quantile(0.25),
        pooled["gain"].quantile(0.75),
        pooled["gain"].min(),
        pooled["gain"].max(),
    )

    logger.info("Step 4: Computing regional exposure index")
    exposure_df = compute_exposure_index(pooled, REGION_NAMES)

    logger.info("\nRegional exposure index:")
    logger.info(
        "\n%s",
        exposure_df[[
            "region", "drgn2", "exposure",
            "mean_gain_pre", "mean_total_post"
        ]].to_string(index=False)
    )
    logger.info("Step 5: Saving outputs")
    save_exposure(exposure_df, pooled, EXPOSURE_OUTPUT_DIR)
    plot_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Exposure variable pipeline complete.")
    logger.info(
        "Outputs saved to: %s", EXPOSURE_OUTPUT_DIR.resolve()
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
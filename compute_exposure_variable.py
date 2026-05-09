"""
compute_exposure.py
===================
Main entry point for the exposure variable pipeline.

Steps:
  1. Load all six EUROMOD output files (3 RMI + 3 IMV)
  2. Validate the IMV counterfactual simulation statistically
  3. Compute three generosity dimensions per region per year
  4. Average dimensions across 2017–2019
  5. Standardize and extract PC1 via PCA → regional exposure index
  6. Save outputs and produce chart

Usage
-----
    python compute_exposure.py

Outputs
-------
    output/exposure/imv_validation_report.csv
    output/exposure/exposure_index.csv
    output/exposure/pca_diagnostics.csv
    output/exposure/pca_loadings.csv
    output/exposure/exposure_index_pca.png
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
    REGION_NAMES,
    RMI_INCOMPATIBLE_REGIONS,
)
from src.exposure_loader import load_all_files
from src.exposure_validation import run_validation
from src.exposure_compute import (
    pool_dimensions,
    compute_pca_exposure,
    plot_exposure,
    save_exposure,
)


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting exposure variable pipeline — PCA approach")
    logger.info("=" * 60)
    logger.info(
        "Excluded regions: La Rioja (23), Aragón (24), Ceuta (63)"
    )
    logger.info(
        "Incompatible regions (bsarg_s_post=0): "
        "Galicia (11), Illes Balears (53), Andalucía (61)"
    )

    # Step 1 — Load files
    logger.info("Step 1: Loading EUROMOD output files")
    rmi_dfs, imv_dfs = load_all_files(RMI_FILES, IMV_FILES)

    # Step 2 — Validate IMV simulation
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
    logger.info("Validation: %d/%d tests passed", passed, total)

    # Step 3 — Compute three generosity dimensions
    logger.info(
        "Step 3: Computing generosity dimensions "
        "(delta_mean, delta_recipients_pc, delta_expenditure_pc)"
    )
    pooled = pool_dimensions(
        rmi_dfs=rmi_dfs,
        imv_dfs=imv_dfs,
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        incompatible_regions=RMI_INCOMPATIBLE_REGIONS,
    )

    logger.info("\nPooled dimensions (averaged 2017-2019):")
    logger.info(
        "\n%s",
        pooled[[
            "drgn2", "delta_mean",
            "delta_recipients_pc", "delta_expenditure_pc"
        ]].to_string(index=False)
    )

    # Step 4 — PCA
    logger.info("Step 4: Extracting PC1 via PCA")
    exposure_df = compute_pca_exposure(pooled, REGION_NAMES)

    logger.info("\nRegional exposure index (PC1):")
    logger.info(
        "\n%s",
        exposure_df[[
            "region", "drgn2", "exposure",
            "delta_mean", "delta_recipients_pc",
            "delta_expenditure_pc"
        ]].to_string(index=False)
    )


    logger.info("Step 5: Saving outputs")
    save_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)
    plot_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Exposure pipeline complete.")
    logger.info("Outputs: %s", EXPOSURE_OUTPUT_DIR.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
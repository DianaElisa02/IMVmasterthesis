"""
compute_exposure_variable.py
===================
Main entry point for the exposure variable pipeline.
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
    residualize_delta_mean,
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
    logger.info("Validation: %d/%d tests passed", passed, total)

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

    logger.info(
        "Step 3.5: Residualizing delta_mean "
        "(partialling out pre-reform RMI level)"
    )
    pooled = residualize_delta_mean(pooled)

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
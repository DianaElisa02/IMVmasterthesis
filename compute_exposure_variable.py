"""
compute_exposure_variable.py
============================
Main entry point for the exposure variable pipeline.

Steps:
  1. Load EUROMOD RMI and IMV simulation outputs
  2. Validate IMV simulation quality and exposure index validity
  3. Compute regional gain dimensions (delta_expenditure_pc, delta_recipients_pc)
     — delta_mean is computed but excluded from PCA (see Step 4)
  4. Extract PC1 via PCA on delta_recipients_pc and delta_expenditure_pc only
  5. Save outputs

Regions excluded throughout:
  - La Rioja (23), Aragón (24): broken EUROMOD RMI parameterisation
  - Ceuta (63), Melilla (64): ECV sample too small for regional estimates

Incompatible regions (bsarg_s zeroed in IMV run):
  - Galicia (11), Illes Balears (53), Andalucía (61)
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
    INFORME_RMI,
    REGION_POPULATION,
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
    logger.info("Starting exposure variable pipeline")
    logger.info("=" * 60)
    logger.info(
        "Excluded regions: La Rioja (23), Aragón (24), Ceuta (63), Melilla (64)"
    )
    logger.info(
        "Incompatible regions (bsarg_s_post=0): "
        "Galicia (11), Illes Balears (53), Andalucía (61)"
    )

    # ------------------------------------------------------------------
    # Step 1: Load EUROMOD outputs
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading EUROMOD output files")
    rmi_dfs, imv_dfs = load_all_files(RMI_FILES, IMV_FILES)

    # ------------------------------------------------------------------
    # Step 2: Compute gain dimensions
    # pool_dimensions returns (pooled_avg, all_dims):
    #   pooled_avg — one row per region, averaged across 2017-2019
    #   all_dims   — one row per region-year, used in stability validation
    # ------------------------------------------------------------------
    logger.info("Step 2: Computing generosity dimensions")
    pooled, all_dims = pool_dimensions(
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

    # ------------------------------------------------------------------
    # Step 3: Extract PC1 via PCA
    # delta_mean is excluded from PCA — it loads negatively against the
    # other two dimensions (R²=0.000 in residualisation, OLS p=0.95),
    # meaning the pre-reform RMI level has no predictive power over the
    # IMV-RMI gain. Including it distorts PC1 without adding signal.
    # It is retained in the output CSV for descriptive transparency.
    # ------------------------------------------------------------------
    logger.info(
        "Step 3: Extracting PC1 via PCA "
        "(delta_recipients_pc and delta_expenditure_pc only — "
        "delta_mean excluded, see module docstring)"
    )
    exposure_df = compute_pca_exposure(
        pooled, REGION_NAMES, dims=["delta_recipients_pc", "delta_expenditure_pc"]
    )

    logger.info("\nRegional exposure index (PC1):")
    logger.info(
        "\n%s",
        exposure_df[[
            "region", "drgn2", "exposure",
            "delta_recipients_pc", "delta_expenditure_pc"
        ]].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Step 4: Validate IMV simulation and exposure index
    # Run after exposure_df is available so institutional consistency
    # tests can use the final PC1 scores.
    # ------------------------------------------------------------------
    logger.info("Step 4: Running validation suite")
    validation_report = run_validation(
        imv_dfs=imv_dfs,
        all_dims=all_dims,
        exposure_df=exposure_df,
        informe_rmi=INFORME_RMI,
        region_population=REGION_POPULATION,
        statutory_single=IMV_STATUTORY_2022["basic_monthly"],
        statutory_max=IMV_STATUTORY_2022["max_monthly"],
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        output_dir=EXPOSURE_OUTPUT_DIR,
    )
    passed = validation_report["pass"].sum()
    total  = validation_report["pass"].notna().sum()
    logger.info("Validation: %d/%d tests passed", passed, total)

    # ------------------------------------------------------------------
    # Step 5: Save outputs
    # ------------------------------------------------------------------
    logger.info("Step 5: Saving outputs")
    save_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)
    plot_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Exposure pipeline complete.")
    logger.info("Outputs: %s", EXPOSURE_OUTPUT_DIR.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
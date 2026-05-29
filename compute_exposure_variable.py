"""
compute_exposure_variable.py
============================
Main entry point for the exposure variable pipeline.

Steps
-----
1. Load EUROMOD RMI and IMV simulation outputs
2. Compute and pool gain dimensions (average before differencing)
3. Construct all exposure specifications
5. Run validation suite
6. Save and plot outputs

Primary DiD regressor is controlled by PRIMARY_SPEC in src/exposure_index.py.
All specifications are always computed and saved regardless.

Regions excluded from ALL specs
---------------------------------
Ceuta (63), Melilla (64): ECV sample too small for regional estimates

Regions excluded from fully simulated specs only
-------------------------------------------------
La Rioja (23), Aragón (24): bsarg_s = €1 placeholder in STD files →
  delta_exp_sim / delta_cov_sim unusable.
  Hybrid and admin specs are clean (RMI side = Informe admin data).

Incompatible regions (bsarg_s zeroed in IMV run)
-------------------------------------------------
Galicia (11), Illes Balears (53), Andalucía (61)
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
    EXPOSURE_SIM_EXCLUDE_REGIONS,        # <-- NEW
    EXPOSURE_OUTPUT_DIR,
    IMV_STATUTORY_2022,
    REGION_NAMES,
    RMI_INCOMPATIBLE_REGIONS,
    INFORME_RMI,
    REGION_POPULATION,
)
from src.exposure_loader import load_all_files
from src.exposure_dimensions import pool_dimensions
from src.exposure_index import PRIMARY_SPEC, compute_exposure
from src.exposure_io import save_exposure, plot_exposure
from src.exposure_validation import run_validation


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting exposure variable pipeline")
    logger.info("Primary specification: %s", PRIMARY_SPEC)
    logger.info("=" * 60)
    logger.info(
        "Excluded from ALL specs: Ceuta (63), Melilla (64)"
    )
    logger.info(
        "Excluded from sim specs only: La Rioja (23), Aragón (24) "
        "(bsarg_s €1 placeholder in STD files)"
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
    # Step 2: Pool dimensions
    # Average raw simulated values across 2017-2019, merge administrative
    # Informe data, then compute all delta dimensions.
    #
    # sim_exclude_regions: La Rioja and Aragón are excluded ONLY from
    # delta_exp_sim and delta_cov_sim. Their hybrid and admin deltas are
    # computed normally using Informe admin data on the RMI side.
    # ------------------------------------------------------------------
    logger.info("Step 2: Pooling dimensions (average before differencing)")
    pooled, all_dims = pool_dimensions(
        rmi_dfs=rmi_dfs,
        imv_dfs=imv_dfs,
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        sim_exclude_regions=EXPOSURE_SIM_EXCLUDE_REGIONS,   # <-- NEW
        incompatible_regions=RMI_INCOMPATIBLE_REGIONS,
        informe_rmi=INFORME_RMI,
        region_population=REGION_POPULATION,
    )

    # ------------------------------------------------------------------
    # Step 3: Construct all exposure specifications
    # ------------------------------------------------------------------
    logger.info(
        "Step 3: Constructing exposure specifications "
        "(primary: %s)", PRIMARY_SPEC
    )
    exposure_df = compute_exposure(pooled, REGION_NAMES)

    # ------------------------------------------------------------------
    # Step 5: Validate IMV simulation and exposure index
    # ------------------------------------------------------------------
    logger.info("Step 5: Running validation suite")
    validation_report = run_validation(
        imv_dfs=imv_dfs,
        all_dims=all_dims,
        exposure_df=exposure_df,
        informe_rmi=INFORME_RMI,
        region_population=REGION_POPULATION,
        statutory_single=IMV_STATUTORY_2022["basic_monthly"],
        statutory_max=IMV_STATUTORY_2022["max_monthly"],
        floor_monthly=IMV_STATUTORY_2022["floor_monthly"],
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        output_dir=EXPOSURE_OUTPUT_DIR,
    )
    passed = validation_report["pass"].sum()
    total  = validation_report["pass"].notna().sum()
    logger.info("Validation: %d/%d tests passed", passed, total)

    # ------------------------------------------------------------------
    # Step 6: Save and plot
    # ------------------------------------------------------------------
    logger.info("Step 6: Saving outputs")
    save_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)
    plot_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Exposure pipeline complete.")
    logger.info("Primary spec:  %s", PRIMARY_SPEC)
    logger.info("Outputs:       %s", EXPOSURE_OUTPUT_DIR.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
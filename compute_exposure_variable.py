"""
compute_exposure_variable.py
============================
Main entry point for the regional exposure-measures pipeline.

Steps
-----
1. Load EUROMOD RMI and IMV simulation outputs
2. Compute and pool regional dimensions (average before differencing)
3. Construct separate exposure specifications
4. Run validation diagnostics
5. Save and plot outputs

Coverage and average annual benefit are analysed as separate co-primary
specifications controlled by PRIMARY_SPECS in src/exposure_index.py.
All specifications are always computed and saved.

Regions excluded from all specifications
----------------------------------------
Ceuta (63), Melilla (64): ECV sample too small for regional estimates.

Regions excluded from fully simulated specifications only
----------------------------------------------------------
La Rioja (23), Aragón (24): bsarg_s = €1 placeholder in STD files, so the
fully simulated pre/post comparison is unusable. Their hybrid and
administrative specifications remain available.
"""

from __future__ import annotations

import logging
import sys

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
    EXPOSURE_EXCLUDE_REGIONS,
    EXPOSURE_OUTPUT_DIR,
    EXPOSURE_SIM_EXCLUDE_REGIONS,
    IMV_FILES,
    IMV_STATUTORY_2022,
    INFORME_RMI,
    REGION_NAMES,
    REGION_POPULATION,
    RMI_FILES,
    RMI_INCOMPATIBLE_REGIONS,
)
from src.exposure_dimensions import pool_dimensions
from src.exposure_index import PRIMARY_SPECS, compute_exposure
from src.exposure_io import plot_exposure, save_exposure
from src.exposure_loader import load_all_files
from src.exposure_validation import run_validation


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting regional exposure-measures pipeline")
    logger.info("Primary specifications: %s", PRIMARY_SPECS)
    logger.info("=" * 60)
    logger.info(
        "Excluded from all specifications: Ceuta (63), Melilla (64)"
    )
    logger.info(
        "Excluded from fully simulated specifications only: "
        "La Rioja (23), Aragón (24) "
        "(bsarg_s €1 placeholder in STD files)"
    )
    logger.info(
        "Incompatible post-reform regional component (bsarg_s_post=0): "
        "Galicia (11), Illes Balears (53), Andalucía (61)"
    )

    logger.info("Step 1: Loading EUROMOD output files")
    rmi_dfs, imv_dfs = load_all_files(RMI_FILES, IMV_FILES)

    logger.info("Step 2: Pooling dimensions (average before differencing)")
    pooled, all_dims = pool_dimensions(
        rmi_dfs=rmi_dfs,
        imv_dfs=imv_dfs,
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        sim_exclude_regions=EXPOSURE_SIM_EXCLUDE_REGIONS,
        incompatible_regions=RMI_INCOMPATIBLE_REGIONS,
        informe_rmi=INFORME_RMI,
        region_population=REGION_POPULATION,
    )

    logger.info(
        "Step 3: Constructing exposure specifications (primary: %s)",
        PRIMARY_SPECS,
    )
    exposure_df = compute_exposure(pooled, REGION_NAMES)

    logger.info("Step 4: Running validation diagnostics")
    run_validation(
        imv_dfs=imv_dfs,
        all_dims=all_dims,
        exposure_df=exposure_df,
        informe_rmi=INFORME_RMI,
        region_population=REGION_POPULATION,
        statutory_single=IMV_STATUTORY_2022["basic_monthly"],
        statutory_max=IMV_STATUTORY_2022["max_monthly"],
        floor_monthly=IMV_STATUTORY_2022["floor_monthly"],
        exclude_regions=EXPOSURE_EXCLUDE_REGIONS,
        sim_exclude_regions=EXPOSURE_SIM_EXCLUDE_REGIONS,
        output_dir=EXPOSURE_OUTPUT_DIR,
    )

    logger.info("Step 5: Saving outputs")
    save_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)
    plot_exposure(exposure_df, EXPOSURE_OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("Exposure-measures pipeline complete.")
    logger.info("Primary specifications: %s", PRIMARY_SPECS)
    logger.info("Outputs: %s", EXPOSURE_OUTPUT_DIR.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

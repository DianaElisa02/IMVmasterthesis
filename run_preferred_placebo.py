"""Run placebo DiD models aligned with the preferred grouped specification.

The placebo sample uses 2017-2019 and assigns a fictitious treatment in 2019.
Low exposure is the reference group; medium and high exposure coefficients are
estimated for the two co-primary hybrid exposure measures using the preferred
demographic controls.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import polars as pl

from src.binned_did import compute_tercile_assignments, run_binned_did_spec
from src.constants import ANALYSIS_OUTCOMES
from src.control_specs import PREFERRED_CONTROLS, add_preferred_control_groups
from src.exposure_specs import EXPOSURE_LABELS, PRIMARY_EXPOSURE_SPECS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "robustness" / "preferred_placebo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLACEBO_YEARS = [2017, 2018, 2019]
FAKE_POST_YEAR = 2019


def build_placebo_data(panel: pl.DataFrame, exposure_spec: str) -> pd.DataFrame:
    panel = add_preferred_control_groups(panel)
    assignments = compute_tercile_assignments(panel, exposure_spec)
    did = (
        panel.filter(pl.col("year").is_in(PLACEBO_YEARS))
        .join(
            pl.from_pandas(assignments[["drgn2", "exposure_tercile"]]),
            on="drgn2",
            how="inner",
        )
        .with_columns(
            pl.col("year").eq(FAKE_POST_YEAR).cast(pl.Float64).alias("post"),
            pl.col("exposure_tercile").eq("medium").cast(pl.Float64).alias("tercile_medium"),
            pl.col("exposure_tercile").eq("high").cast(pl.Float64).alias("tercile_high"),
        )
        .with_columns(
            (pl.col("post") * pl.col("tercile_medium")).alias("post_x_medium"),
            (pl.col("post") * pl.col("tercile_high")).alias("post_x_high"),
        )
    )
    return did.to_pandas()


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def main() -> None:
    panel = pl.read_parquet(INPUT_PATH)
    rows: list[dict] = []

    for spec_i, exposure_spec in enumerate(PRIMARY_EXPOSURE_SPECS):
        df = build_placebo_data(panel, exposure_spec)
        missing = [control for control in PREFERRED_CONTROLS if control not in df.columns]
        if missing:
            raise ValueError(f"Preferred controls missing from placebo data: {missing}")

        for outcome_i, outcome in enumerate(ANALYSIS_OUTCOMES):
            seed = 5000 + spec_i * 100 + outcome_i * 2
            result = run_binned_did_spec(
                df=df,
                outcome=outcome,
                controls=PREFERRED_CONTROLS,
                seed_medium=seed,
                seed_high=seed + 1,
            )
            result.update(
                {
                    "exposure_spec": exposure_spec,
                    "control_spec": "preferred_demographic",
                    "placebo_post_year": FAKE_POST_YEAR,
                }
            )
            rows.append(result)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "preferred_tercile_placebo.csv", index=False)

    print("\nPREFERRED TERCILE PLACEBO — FAKE TREATMENT IN 2019")
    for _, row in results.iterrows():
        print(
            f"{EXPOSURE_LABELS.get(row['exposure_spec'], row['exposure_spec'])} | "
            f"{row['outcome']} | medium={row['coef_medium']:+.4f} "
            f"(p_WCB={row['pval_wbt_medium']:.4f}{_stars(row['pval_wbt_medium'])}) | "
            f"high={row['coef_high']:+.4f} "
            f"(p_WCB={row['pval_wbt_high']:.4f}{_stars(row['pval_wbt_high'])})"
        )

    logger.info("Preferred placebo output saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

"""Compare unadjusted and alternative control sets in the baseline tercile DiD."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.binned_did import build_binned_did_data, run_binned_did_spec
from src.constants import ANALYSIS_OUTCOMES, DID_POST_YEARS_BASELINE, DID_POST_YEARS_COVID
from src.exposure_specs import PRIMARY_EXPOSURE_SPECS, EXPOSURE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "binned_did" / "control_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONTROL_SPECS: dict[str, list[str]] = {
    "unadjusted": [],
    "preferred_demographic": [
        "head_age_group",
        "head_sex",
        "head_education_group",
        "n_adults_group",
        "n_children_group",
    ],
    "extended_composition": [
        "head_age_group",
        "head_sex",
        "head_education_group",
        "n_adults_group",
        "n_children_group",
        "single_parent_hh",
        "homeowner",
    ],
    "legacy_full": [
        "hh_size",
        "n_children",
        "head_age_group",
        "head_sex",
        "head_labour_group",
        "single_parent_hh",
        "homeowner",
    ],
}


def _add_grouped_composition(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(
        pl.when(pl.col("n_adults").eq(1)).then(pl.lit("1"))
        .when(pl.col("n_adults").eq(2)).then(pl.lit("2"))
        .when(pl.col("n_adults").ge(3)).then(pl.lit("3plus"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("n_adults_group"),
        pl.when(pl.col("n_children").eq(0)).then(pl.lit("0"))
        .when(pl.col("n_children").eq(1)).then(pl.lit("1"))
        .when(pl.col("n_children").eq(2)).then(pl.lit("2"))
        .when(pl.col("n_children").ge(3)).then(pl.lit("3plus"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("n_children_group"),
    )


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def _print_results(results: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("TERCILE DiD CONTROL-SENSITIVITY RESULTS")
    print("=" * 120)
    for _, row in results.iterrows():
        print(
            f"{row['period']} | {row['sample_rule']} | {row['control_spec']} | "
            f"{EXPOSURE_LABELS[row['exposure_spec']]} | {row['outcome']} | "
            f"N={int(row['n_obs'])} | "
            f"medium={row['coef_medium']:+.4f} (p_WCB={row['pval_wbt_medium']:.4f}{_stars(row['pval_wbt_medium'])}) | "
            f"high={row['coef_high']:+.4f} (p_WCB={row['pval_wbt_high']:.4f}{_stars(row['pval_wbt_high'])})"
        )


def main() -> None:
    panel = _add_grouped_composition(pl.read_parquet(INPUT_PATH))
    all_rows: list[dict] = []

    for period, post_years in [
        ("baseline_2021_2025", DID_POST_YEARS_BASELINE),
        ("covid_robust_2022_2025", DID_POST_YEARS_COVID),
    ]:
        for spec_i, exposure_spec in enumerate(PRIMARY_EXPOSURE_SPECS):
            did, _ = build_binned_did_data(panel, post_years, exposure_spec)
            df = did.to_pandas()

            preferred_complete_cols = CONTROL_SPECS["extended_composition"]
            common_mask = df[preferred_complete_cols].notna().all(axis=1)

            for sample_rule, sample_df in [
                ("available_case", df),
                ("common_sample", df.loc[common_mask].copy()),
            ]:
                for control_i, (control_name, controls) in enumerate(CONTROL_SPECS.items()):
                    missing = [col for col in controls if col not in sample_df.columns]
                    if missing:
                        logger.warning("Skipping %s; missing columns: %s", control_name, missing)
                        continue
                    for outcome_i, outcome in enumerate(ANALYSIS_OUTCOMES):
                        if outcome not in sample_df.columns:
                            continue
                        seed_base = 1000 + spec_i * 500 + control_i * 50 + outcome_i * 2
                        result = run_binned_did_spec(
                            sample_df,
                            outcome,
                            controls,
                            seed_medium=seed_base,
                            seed_high=seed_base + 1,
                        )
                        result.update(
                            {
                                "period": period,
                                "sample_rule": sample_rule,
                                "control_spec": control_name,
                                "exposure_spec": exposure_spec,
                                "common_sample_n_before_outcome_missing": int(len(sample_df)),
                            }
                        )
                        all_rows.append(result)

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "binned_did_control_sensitivity.csv", index=False)

    comparison_cols = [
        "period",
        "sample_rule",
        "control_spec",
        "exposure_spec",
        "outcome",
        "n_obs",
        "coef_medium",
        "pval_wbt_medium",
        "coef_high",
        "pval_wbt_high",
        "controls",
    ]
    results[comparison_cols].to_csv(
        OUTPUT_DIR / "binned_did_control_sensitivity_compact.csv", index=False
    )

    _print_results(results)
    logger.info("Control-sensitivity outputs saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()

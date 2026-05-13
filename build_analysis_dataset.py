"""
build_analysis_dataset.py

"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.constants import ANALYSIS_YEARS, EXPOSURE_SPECS
from src.ecv_clean import build_analysis_panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


BASE_PATH = Path("/workspaces/IMVmasterthesis/input_data")
EXPOSURE_PATH = Path("/workspaces/IMVmasterthesis") / "output" / "exposure" / "exposure_index.csv"
OUTPUT_DIR    = BASE_PATH / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_OUTPUT = OUTPUT_DIR / "analysis_dataset.parquet"
CHECKS_OUTPUT   = OUTPUT_DIR / "dataset_checks.csv"


# =============================================================================
# DESCRIPTIVE CHECKS
# =============================================================================

def _weighted_mean(col: str, weight: str, df: pl.DataFrame) -> float:
    sub = df.select([col, weight]).drop_nulls()
    sub = sub.filter(pl.col(weight).gt(0))
    if sub.is_empty():
        return float("nan")
    vals = sub[col].to_numpy()
    wts  = sub[weight].to_numpy()
    return float((vals * wts).sum() / wts.sum())


def make_checks(panel: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for year in sorted(panel["year"].unique().to_list()):
        g = panel.filter(pl.col("year").eq(year))
        rows.append({
            "year":                   year,
            "n_households":           len(g),
            "n_regions":              g["drgn2"].n_unique(),
            "matdep_rate_pct":        100 * _weighted_mean("matdep",  "weight_hh", g),
            "poverty_rate_pct":       100 * _weighted_mean("poverty", "weight_hh", g),
            "mean_income_net_annual": _weighted_mean("income_net_annual", "weight_hh", g),
            "mean_hh_size":           _weighted_mean("hh_size", "weight_hh", g),
            "pct_matdep_missing":     100 * g.filter(pl.col("matdep").is_null()).height / len(g),
            "pct_poverty_missing":    100 * g.filter(pl.col("poverty").is_null()).height / len(g),
            "pct_income_missing":     100 * g.filter(pl.col("income_net_annual").is_null()).height / len(g),
            "pct_head_age_available": 100 * g.filter(pl.col("head_age").is_not_null()).height / len(g),
        })
    return pl.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("=== IMV DiD — build_analysis_dataset.py ===")
    logger.info("Years: %s", ANALYSIS_YEARS)

    panel = build_analysis_panel(
        years=ANALYSIS_YEARS,
        ecv_dir=BASE_PATH,
        exposure_path=EXPOSURE_PATH,
    )

    checks = make_checks(panel)

    print("\n=== Year-by-year checks ===")
    print(checks)

    panel.write_parquet(ANALYSIS_OUTPUT)
    checks.write_csv(CHECKS_OUTPUT)

    logger.info("Saved: %s  (%d rows)", ANALYSIS_OUTPUT, len(panel))
    logger.info("Saved: %s", CHECKS_OUTPUT)

    pre  = panel.filter(pl.col("post").eq(0.0)).height
    post = panel.filter(pl.col("post").eq(1.0)).height
    print(f"\n=== Final dataset ===")
    print(f"  Household-year obs   : {len(panel):,}")
    print(f"  Years                : {sorted(panel['year'].unique().to_list())}")
    print(f"  Regions (clusters)   : {panel['drgn2'].n_unique()}")
    print(f"  Pre-reform  (post=0) : {pre:,}")
    print(f"  Post-reform (post=1) : {post:,}")
    print(f"  matdep missing       : {panel.filter(pl.col('matdep').is_null()).height:,}")
    print(f"  poverty missing      : {panel.filter(pl.col('poverty').is_null()).height:,}")
    primary = EXPOSURE_SPECS[0]
    if primary in panel.columns:
        matched = panel.filter(pl.col(primary).is_not_null()).height
        print(f"  Exposure matched     : {matched:,}")
    print(f"\nColumns:\n  {sorted(panel.columns)}")


if __name__ == "__main__":
    main()
"""
run_balance_checks.py
=====================
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.balance_checks import run_balance_checks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_PATH   = Path("/workspaces/IMVmasterthesis")
INPUT_PATH  = BASE_PATH / "output" / "analysis_dataset.parquet"
OUTPUT_DIR  = BASE_PATH / "output" / "balance_checks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_OUTPUT = OUTPUT_DIR / "balance_table.csv"

def main() -> None:
    logger.info("=== IMV DiD — run_balance_checks.py ===")

    panel = pl.read_parquet(INPUT_PATH)
    logger.info("Panel loaded: %d obs", len(panel))

    table = run_balance_checks(panel)

    table.write_csv(TABLE_OUTPUT)
    logger.info("Saved: %s", TABLE_OUTPUT)

    print("\n=== Pre-reform balance table (weighted means by exposure tercile) ===")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=4):
        print(table)


if __name__ == "__main__":
    main()
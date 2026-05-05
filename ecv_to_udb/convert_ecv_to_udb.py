"""
convert_ecv_to_udb.py
=====================
Entry point for the ECV → EUROMOD UDB conversion pipeline.

For each survey year (2017, 2018, 2019), reads the raw ECV Td, Th, Tr, Tp
files from input_data/, builds household and person-level UDB variables,
merges them, validates, and exports a tab-separated .txt file to output/
ready for loading into EUROMOD.

Usage
-----
    python convert_ecv_to_udb.py

Output files
------------
    output/ES_2017_a2.txt
    output/ES_2018_a1.txt
    output/ES_2019_b1.txt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
import np

from src.constants import EUROMOD_DATASET_NAMES, YEARS
from src.household import build_household_udb
from src.merge import merge_and_export
from src.person import build_person_udb

BASE_DIR   = Path(__file__).resolve().parent
INPUT_DIR  = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR / "output"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "conversion.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting ECV → EUROMOD UDB conversion")
    logger.info("Input  : %s", INPUT_DIR)
    logger.info("Output : %s", OUTPUT_DIR)

    failed: list[int] = []

    for year in YEARS:
        logger.info("=" * 60)
        logger.info("Processing year %s", year)
        try:
            household_udb = build_household_udb(INPUT_DIR, year)
            person_udb    = build_person_udb(INPUT_DIR, year)

            dataset_name = EUROMOD_DATASET_NAMES[year]
            output_path  = OUTPUT_DIR / f"{dataset_name}.txt"

            merge_and_export(person_udb, household_udb, output_path, year)

        except FileNotFoundError as exc:
            logger.error("Year %s: missing input file — %s", year, exc)
            failed.append(year)
        except Exception as exc:
            logger.error("Year %s: conversion failed — %s", year, exc, exc_info=True)
            failed.append(year)

    logger.info("=" * 60)
    if failed:
        logger.error("Conversion completed with failures for years: %s", failed)
        sys.exit(1)
    else:
        logger.info("Conversion completed successfully for all years: %s", YEARS)


if __name__ == "__main__":
    main()
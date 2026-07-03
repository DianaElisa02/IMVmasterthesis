"""Recover ECV household cross-sectional weights from the raw Td files.

The analysis parquet currently does not retain the household weight. This helper
searches recursively under ``input_data`` for files named ``ECV_Td_YYYY.dta``,
extracts DB030 (household identifier) and DB090 (cross-sectional household
weight), and writes a compact household-year weight file that can be merged by
``make_descriptive_statistics.py``.

Run:
    python prepare_ecv_household_weights.py
    python make_descriptive_statistics.py \
        --weights output/descriptive_statistics/ecv_household_weights.parquet
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import polars as pl

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "input_data"
DEFAULT_OUTPUT = (
    BASE_DIR / "output" / "descriptive_statistics" / "ecv_household_weights.parquet"
)
FILE_RE = re.compile(r"ECV_Td_(\d{4})\.dta$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalise_household_id(series: pd.Series) -> pd.Series:
    """Match the identifier construction used by the UDB household builder."""
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.astype("string")


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    files = sorted(
        path
        for path in args.input_dir.rglob("*.dta")
        if FILE_RE.search(path.name)
    )
    if not files:
        raise FileNotFoundError(
            f"No ECV_Td_YYYY.dta files found recursively under {args.input_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in files:
        match = FILE_RE.search(path.name)
        assert match is not None
        year = int(match.group(1))

        raw = pd.read_stata(
            path,
            columns=["DB030", "DB090"],
            convert_categoricals=False,
        )
        raw.columns = [column.upper() for column in raw.columns]
        if "DB030" not in raw.columns or "DB090" not in raw.columns:
            raise ValueError(f"{path} does not contain DB030 and DB090")

        block = pd.DataFrame(
            {
                "year": year,
                "idhh": normalise_household_id(raw["DB030"]),
                "dwt": pd.to_numeric(raw["DB090"], errors="coerce"),
            }
        )
        block = block.dropna(subset=["idhh", "dwt"])
        block = block.loc[block["dwt"] > 0]

        duplicated = block.duplicated(["year", "idhh"], keep=False)
        if duplicated.any():
            conflicts = (
                block.loc[duplicated]
                .groupby(["year", "idhh"], as_index=False)["dwt"]
                .nunique()
            )
            if (conflicts["dwt"] > 1).any():
                raise ValueError(
                    f"Conflicting DB090 values found for duplicate household IDs in {path}"
                )
            block = block.drop_duplicates(["year", "idhh"], keep="first")

        frames.append(block)
        print(f"{year}: extracted {len(block):,} household weights from {path}")

    weights = pd.concat(frames, ignore_index=True)
    weights = weights.sort_values(["year", "idhh"]).reset_index(drop=True)

    duplicated = weights.duplicated(["year", "idhh"], keep=False)
    if duplicated.any():
        examples = weights.loc[duplicated, ["year", "idhh"]].head(10)
        raise ValueError(
            "Duplicate household-year identifiers remain after combining files:\n"
            + examples.to_string(index=False)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(weights).write_parquet(args.output)

    print("\nWeight file created successfully")
    print(f"Years: {sorted(weights['year'].unique().tolist())}")
    print(f"Household-year records: {len(weights):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()

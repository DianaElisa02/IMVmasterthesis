"""Inspect candidate control variables in the final analysis parquet.

Outputs:
- control_variable_summary.csv
- control_missingness_by_year.csv
- control_weighted_frequencies.csv
- control_unique_values.csv

The script also prints a compact terminal summary.
"""
from __future__ import annotations

from pathlib import Path
import math
import pandas as pd
import polars as pl

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "output" / "analysis_dataset_with_gap.parquet"
OUTPUT_DIR = BASE_DIR / "output" / "control_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_VARIABLES = [
    "head_age",
    "head_age_group",
    "head_sex",
    "head_education_group",
    "head_education_isced",
    "head_high_education",
    "n_adults",
    "n_children",
    "hh_size",
    "single_parent_hh",
    "head_non_eu_born",
    "homeowner",
    "head_labour_group",
    "head_unemployed",
    "head_employed",
    "DB100",
    "drgmd",
    "drgru",
    "drgur",
]

WEIGHT_CANDIDATES = ["dwt", "DB090", "weight", "survey_weight"]
COUNT_VARIABLES = {"head_age", "head_education_isced", "n_adults", "n_children", "hh_size"}


def _detect_weight(columns: list[str]) -> str | None:
    return next((col for col in WEIGHT_CANDIDATES if col in columns), None)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return math.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def _format_value(value: object) -> str:
    if pd.isna(value):
        return "<MISSING>"
    return str(value)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Analysis dataset not found: {INPUT_PATH}")

    panel = pl.read_parquet(INPUT_PATH)
    columns = panel.columns
    available = [col for col in CANDIDATE_VARIABLES if col in columns]
    absent = [col for col in CANDIDATE_VARIABLES if col not in columns]
    weight_col = _detect_weight(columns)

    if "year" not in columns:
        raise ValueError("The analysis dataset must contain a 'year' column")

    selected = ["year"] + available + ([weight_col] if weight_col else [])
    df = panel.select(selected).to_pandas()

    summary_rows: list[dict] = []
    missing_rows: list[dict] = []
    frequency_rows: list[dict] = []
    unique_rows: list[dict] = []

    for variable in available:
        series = df[variable]
        nonmissing = series.dropna()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        n_total = len(series)
        n_missing = int(series.isna().sum())
        n_unique = int(nonmissing.nunique(dropna=True))

        minimum = float(nonmissing.min()) if is_numeric and len(nonmissing) else math.nan
        maximum = float(nonmissing.max()) if is_numeric and len(nonmissing) else math.nan
        mean = float(nonmissing.mean()) if is_numeric and len(nonmissing) else math.nan
        weighted_mean = (
            _weighted_mean(pd.to_numeric(series, errors="coerce"), pd.to_numeric(df[weight_col], errors="coerce"))
            if is_numeric and weight_col
            else math.nan
        )

        summary_rows.append({
            "variable": variable,
            "dtype": str(series.dtype),
            "classification": (
                "count_or_continuous" if variable in COUNT_VARIABLES
                else "binary" if n_unique <= 2
                else "categorical"
            ),
            "n_total": n_total,
            "n_nonmissing": int(series.notna().sum()),
            "n_missing": n_missing,
            "missing_pct": 100 * n_missing / n_total if n_total else math.nan,
            "n_unique_nonmissing": n_unique,
            "min": minimum,
            "max": maximum,
            "unweighted_mean": mean,
            "weighted_mean": weighted_mean,
            "weight_column": weight_col or "none",
        })

        for year, block in df.groupby("year", dropna=False):
            n_year = len(block)
            n_missing_year = int(block[variable].isna().sum())
            missing_rows.append({
                "variable": variable,
                "year": year,
                "n_total": n_year,
                "n_missing": n_missing_year,
                "missing_pct": 100 * n_missing_year / n_year if n_year else math.nan,
            })

        values = series.drop_duplicates().sort_values(na_position="last")
        for value in values.tolist():
            unique_rows.append({
                "variable": variable,
                "value": _format_value(value),
                "is_missing": bool(pd.isna(value)),
            })

        frequency_source = df[[variable] + ([weight_col] if weight_col else [])].copy()
        frequency_source["value"] = frequency_source[variable].map(_format_value)
        if weight_col:
            frequency_source["_weight"] = pd.to_numeric(frequency_source[weight_col], errors="coerce").fillna(0.0)
        else:
            frequency_source["_weight"] = 1.0

        grouped = (
            frequency_source.groupby("value", dropna=False)
            .agg(unweighted_n=(variable, "size"), weighted_n=("_weight", "sum"))
            .reset_index()
        )
        total_weight = grouped["weighted_n"].sum()
        total_n = grouped["unweighted_n"].sum()
        grouped["unweighted_pct"] = 100 * grouped["unweighted_n"] / total_n if total_n else math.nan
        grouped["weighted_pct"] = 100 * grouped["weighted_n"] / total_weight if total_weight else math.nan
        grouped.insert(0, "variable", variable)
        frequency_rows.extend(grouped.to_dict("records"))

    summary = pd.DataFrame(summary_rows)
    missing = pd.DataFrame(missing_rows)
    frequencies = pd.DataFrame(frequency_rows)
    unique_values = pd.DataFrame(unique_rows)

    summary.to_csv(OUTPUT_DIR / "control_variable_summary.csv", index=False)
    missing.to_csv(OUTPUT_DIR / "control_missingness_by_year.csv", index=False)
    frequencies.to_csv(OUTPUT_DIR / "control_weighted_frequencies.csv", index=False)
    unique_values.to_csv(OUTPUT_DIR / "control_unique_values.csv", index=False)

    print("\n" + "=" * 110)
    print("CANDIDATE CONTROL VARIABLE DIAGNOSTICS")
    print("=" * 110)
    print(f"Input: {INPUT_PATH}")
    print(f"Weight column: {weight_col or 'none found — frequencies are unweighted'}")
    print(f"Available candidate variables ({len(available)}): {available}")
    print(f"Absent candidate variables ({len(absent)}): {absent}")

    if not summary.empty:
        display_cols = [
            "variable", "dtype", "classification", "n_unique_nonmissing",
            "missing_pct", "min", "max", "weighted_mean",
        ]
        print("\nOVERALL SUMMARY")
        print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nMISSINGNESS BY YEAR")
    for variable in available:
        block = missing.loc[missing["variable"] == variable, ["year", "missing_pct"]]
        values = ", ".join(f"{int(row.year)}={row.missing_pct:.1f}%" for row in block.itertuples())
        print(f"{variable}: {values}")

    print("\nWEIGHTED FREQUENCIES")
    for variable in available:
        block = frequencies.loc[frequencies["variable"] == variable].sort_values("weighted_pct", ascending=False)
        print(f"\n{variable}")
        print(block[["value", "unweighted_n", "unweighted_pct", "weighted_n", "weighted_pct"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nDiagnostics saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import numpy as np


REPO = Path("/workspaces/IMVmasterthesis")

INPUT_FILES = {
    2017: REPO / "output" / "ES_2017_a2.txt",
    2018: REPO / "output" / "ES_2018_a1.txt",
    2019: REPO / "output" / "ES_2019_b1.txt",
}

EUROMOD_OUTPUT_FILES = {
    2017: REPO / "input_data" / "euromod_output" / "IMV_2022ruleson2017.txt",
    2018: REPO / "input_data" / "euromod_output" / "IMV_2022ruleson2018.txt",
    2019: REPO / "input_data" / "euromod_output" / "IMV_2022ruleson2019.txt",
}

INCOME_VARS = [
    "yem", "yse", "ypp", "kfb", "kfbcc",
    "bun", "bhl", "pdi", "poa", "psu",
    "bed", "tscer", "xpp",
]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def normalise_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "idhh":
            rename_map[col] = "IDHH"
        if lower in {"idperson", "idpers"}:
            rename_map[col] = "idperson"

    df = df.rename(columns=rename_map)

    if "IDHH" in df.columns:
        df["IDHH"] = df["IDHH"].astype(str)
    if "idperson" in df.columns:
        df["idperson"] = df["idperson"].astype(str)

    return df


def compare_variable(inp: pd.DataFrame, out: pd.DataFrame, var: str) -> dict:
    result = {
        "variable": var,
        "status": "",
        "n_positive_input": np.nan,
        "median_ratio": np.nan,
        "mean_ratio": np.nan,
        "p01_ratio": np.nan,
        "p99_ratio": np.nan,
        "share_ratio_equal_1": np.nan,
    }

    if var not in inp.columns:
        result["status"] = "missing from input"
        return result

    if var not in out.columns:
        result["status"] = "missing from EUROMOD output"
        return result

    id_cols = ["IDHH", "idperson"]
    if not all(c in inp.columns for c in id_cols) or not all(c in out.columns for c in id_cols):
        result["status"] = "ID columns missing"
        return result

    comp = inp[id_cols + [var]].merge(
        out[id_cols + [var]],
        on=id_cols,
        how="inner",
        suffixes=("_input", "_output"),
    )

    if comp.empty:
        result["status"] = "no matched rows"
        return result

    comp[f"{var}_input"] = pd.to_numeric(comp[f"{var}_input"], errors="coerce")
    comp[f"{var}_output"] = pd.to_numeric(comp[f"{var}_output"], errors="coerce")

    nonzero = comp[
        comp[f"{var}_input"].notna()
        & comp[f"{var}_output"].notna()
        & (comp[f"{var}_input"] != 0)
    ].copy()

    result["n_positive_input"] = len(nonzero)

    if nonzero.empty:
        result["status"] = "no positive input observations"
        return result

    nonzero["ratio"] = nonzero[f"{var}_output"] / nonzero[f"{var}_input"]

    result["status"] = "compared"
    result["median_ratio"] = nonzero["ratio"].median()
    result["mean_ratio"] = nonzero["ratio"].mean()
    result["p01_ratio"] = nonzero["ratio"].quantile(0.01)
    result["p99_ratio"] = nonzero["ratio"].quantile(0.99)
    result["share_ratio_equal_1"] = np.isclose(nonzero["ratio"], 1.0, rtol=1e-8, atol=1e-8).mean()

    return result


def main() -> None:
    all_results = []

    for year in [2017, 2018, 2019]:
        print(f"\n=== Checking {year} ===")

        inp = normalise_id_columns(read_table(INPUT_FILES[year]))
        out = normalise_id_columns(read_table(EUROMOD_OUTPUT_FILES[year]))

        print(f"Input file:  {INPUT_FILES[year]}")
        print(f"Output file: {EUROMOD_OUTPUT_FILES[year]}")
        print(f"Input shape:  {inp.shape}")
        print(f"Output shape: {out.shape}")

        available_input = [v for v in INCOME_VARS if v in inp.columns]
        available_output = [v for v in INCOME_VARS if v in out.columns]

        print(f"Income variables in input:  {available_input}")
        print(f"Income variables in output: {available_output}")

        for var in INCOME_VARS:
            res = compare_variable(inp, out, var)
            res["year"] = year
            all_results.append(res)

            if res["status"] == "compared":
                print(
                    f"{var:8s} | median ratio: {res['median_ratio']:.4f} | "
                    f"mean ratio: {res['mean_ratio']:.4f} | "
                    f"share exactly 1: {res['share_ratio_equal_1']:.3f}"
                )
            else:
                print(f"{var:8s} | {res['status']}")

    results = pd.DataFrame(all_results)
    output_path = REPO / "output" / "euromod_uprating_check.csv"
    results.to_csv(output_path, index=False)

    print("\n=== Summary ===")
    compared = results[results["status"] == "compared"].copy()

    if compared.empty:
        print("No income variables could be compared between input and EUROMOD output.")
        print("This probably means the EUROMOD output does not export the original income variables.")
    else:
        print(
            compared[
                [
                    "year",
                    "variable",
                    "n_positive_input",
                    "median_ratio",
                    "mean_ratio",
                    "share_ratio_equal_1",
                ]
            ].to_string(index=False)
        )

        if (compared["share_ratio_equal_1"] > 0.99).all():
            print("\nInterpretation: income variables appear not to have been uprated.")
        elif (compared["median_ratio"] > 1.01).any():
            print("\nInterpretation: at least some income variables appear to have been uprated.")
        else:
            print("\nInterpretation: results are mixed; inspect the CSV manually.")

    print(f"\nSaved diagnostic table to: {output_path}")


if __name__ == "__main__":
    main() 
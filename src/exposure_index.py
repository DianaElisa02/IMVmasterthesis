"""
exposure_index.py
=================
Constructs all exposure specifications from pooled dimensions.

Specification selector
----------------------
Change PRIMARY_SPEC to switch the primary DiD regressor.
All specifications are always computed and saved to exposure_index.csv.

Valid values for PRIMARY_SPEC:
    "exposure_composite_hybrid" — hybrid composite (DEFAULT)
    "exposure_exp_hybrid"       — hybrid expenditure only
    "exposure_cov_hybrid"       — hybrid coverage only
    "exposure_composite_sim"    — fully simulation-based composite
    "exposure_admin"            — purely administrative (no simulation)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# ── Specification selector ─────────────────────────────────────────────────
# Change this single value to switch the primary DiD regressor.
# All five specifications are always computed and saved regardless.
PRIMARY_SPEC: str = "exposure_composite_hybrid"

# ── Specification definitions ──────────────────────────────────────────────
# Each entry: (output_column, input_dimensions, weights, description)
# weights: list of floats summing to 1, one per dimension
SPECS: list[dict] = [
    {
        "name":        "exposure_composite_hybrid",
        "dims":        ["delta_exp_hybrid", "delta_cov_hybrid"],
        "weights":     [0.5, 0.5],
        "description": "Hybrid composite — simulated IMV vs administrative RMI "
                       "(expenditure + coverage, equally weighted)",
        "primary":     True,
    },
    {
        "name":        "exposure_exp_hybrid",
        "dims":        ["delta_exp_hybrid"],
        "weights":     [1.0],
        "description": "Hybrid expenditure only — simulated IMV exp vs "
                       "administrative RMI exp",
        "primary":     False,
    },
    {
        "name":        "exposure_cov_hybrid",
        "dims":        ["delta_cov_hybrid"],
        "weights":     [1.0],
        "description": "Hybrid coverage only — simulated IMV recipients vs "
                       "administrative RMI titulares",
        "primary":     False,
    },
    {
        "name":        "exposure_composite_sim",
        "dims":        ["delta_exp_sim", "delta_cov_sim"],
        "weights":     [0.5, 0.5],
        "description": "Fully simulated composite — both sides from EUROMOD "
                       "(expenditure + coverage, equally weighted)",
        "primary":     False,
    },
    {
        "name":        "exposure_admin",
        "dims":        ["delta_exp_admin", "delta_cov_admin"],
        "weights":     [0.5, 0.5],
        "description": "Purely administrative — negative pre-reform RMI "
                       "expenditure + coverage intensity (no simulation)",
        "primary":     False,
    },
]

def _standardise(series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Scale series by std only — mean is NOT removed.
    Zero point preserved: zero = no net change in protection from reform.
    Returns (scaled_series, raw_mean, std).
    """
    mean_ = series.mean()
    std_  = series.std(ddof=1)
    return series / std_, mean_, std_

def compute_exposure(
    pooled: pd.DataFrame,
    region_names: dict[int, str],
) -> pd.DataFrame:
    """
    Construct all exposure specifications from pooled dimensions.

    For each specification:
      1. Scale each input dimension by std only — mean NOT removed.
         Zero point is preserved: zero = no net change in protection
         from the reform. Regions with positive scores genuinely gained;
         regions with negative scores saw a net reduction relative to
         their pre-existing scheme.
      2. Compute weighted average of scaled dimensions.
      3. Add rank version (1=lowest exposure, N=highest).

    Scaling parameters stored in df.attrs for reproducibility.

    Parameters
    ----------
    pooled       : output of pool_dimensions — one row per region.
    region_names : mapping from drgn2 to region name string.

    Returns
    -------
    pd.DataFrame sorted by PRIMARY_SPEC descending.
    df.attrs contains scaling parameters for all dimensions.
    """
    result = pooled.copy()
    result["region"] = result["drgn2"].map(region_names)

    # Collect scaling params for all dimensions used
    std_params: dict[str, dict[str, float]] = {}

    for spec in SPECS:
        z_cols = []
        for dim in spec["dims"]:
            z_col = f"_z_{dim}"
            z_series, mean_, std_ = _standardise(result[dim])
            result[z_col] = z_series
            std_params[dim] = {
                "raw_mean": round(mean_, 6),  # stored for reference only
                "std":      round(std_,  6),
            }
            z_cols.append(z_col)

        # Weighted average of scaled dimensions
        weights = spec["weights"]
        result[spec["name"]] = sum(
            w * result[z] for w, z in zip(weights, z_cols)
        ).round(4)

        # Rank version (1 = lowest exposure, N = highest)
        result[f"{spec['name']}_rank"] = rankdata(
            result[spec["name"]]
        ).astype(int)

        logger.info(
            "Spec %-35s | dims: %s | weights: %s",
            spec["name"], spec["dims"], spec["weights"],
        )

    # Drop intermediate z-columns — internal computation only
    z_temp_cols = [c for c in result.columns if c.startswith("_z_")]
    result = result.drop(columns=z_temp_cols)

    # Store scaling params in attrs
    result.attrs["std_params"]   = std_params
    result.attrs["primary_spec"] = PRIMARY_SPEC

    # Log regional ordering by primary spec
    display  = result.sort_values(PRIMARY_SPEC, ascending=False)
    spec_cols = [s["name"] for s in SPECS]
    logger.info(
        "\nRegional exposure index (sorted by %s):\n%s",
        PRIMARY_SPEC,
        display[["region", "drgn2"] + spec_cols].to_string(index=False),
    )

    return display.reset_index(drop=True)

def _identify_reference_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the household reference person per idhh.
    Uses HB080/HB090 if available; falls back to oldest adult.
    Returns df with boolean column 'is_reference'.
    """
    # Oldest adult per household as proxy for reference person
    adults = df[df["dag"] >= 18].copy()
    ref = (
        adults.sort_values("dag", ascending=False)
        .drop_duplicates(subset="idhh")
        .set_index("idhh")["idperson"]
        .rename("ref_person")
    )
    df = df.copy()
    df["is_reference"] = (
        df.set_index("idhh")["idperson"]
        .eq(ref)
        .reset_index(drop=True)
        .values
    )
    return df


def _filter_group(
    df: pd.DataFrame,
    group_label: str,
) -> pd.DataFrame:
    """
    Filter a person-level dataframe to households belonging to group g.
    Returns the filtered person-level dataframe.
    """
    if group_label == "unemployed_head":
        # Households where the reference person is unemployed (les=5)
        df = _identify_reference_person(df)
        ref_unemployed = df[
            df["is_reference"] & (df["les"] == 5)
        ]["idhh"].unique()
        return df[df["idhh"].isin(ref_unemployed)].copy()

    elif group_label == "single_parent":
        # Households with exactly one adult and at least one child (dag < 18)
        hh_stats = df.groupby("idhh").apply(
            lambda x: pd.Series({
                "n_adults":   (x["dag"] >= 18).sum(),
                "n_children": (x["dag"] < 18).sum(),
            })
        ).reset_index()
        single_parent_hh = hh_stats[
            (hh_stats["n_adults"] == 1) & (hh_stats["n_children"] >= 1)
        ]["idhh"].unique()
        return df[df["idhh"].isin(single_parent_hh)].copy()

    elif group_label == "low_income":
        # Households with yds below national median
        # Deduplicate to household level for median computation
        hh_yds = df.drop_duplicates(subset="idhh")["yds"]
        national_median = hh_yds.median()
        low_income_hh = (
            df.drop_duplicates(subset="idhh")
            .loc[df.drop_duplicates(subset="idhh")["yds"] <= national_median, "idhh"]
            .unique()
        )
        return df[df["idhh"].isin(low_income_hh)].copy()

    else:
        raise ValueError(
            f"Unknown group label: '{group_label}'. "
            f"Valid labels: {[g['label'] for g in GROUPS]}"
        )


def compute_group_exposure(
    rmi_dfs: dict[int, pd.DataFrame],
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
    region_names: dict[int, str],
) -> pd.DataFrame:
    """
    Compute simulation-based mean benefit gain for each group × region.

    For each group g and region r:
        exposure_group_r,g = avg_imv_mean_r,g - avg_rmi_mean_r,g

    Both sides from the simulation — this is where the RMI simulation
    uniquely contributes because the Informe has no household-type breakdown.
    Average before differencing: means averaged across 2017-2019 per group
    per region, then differenced.

    To add or change groups, edit GROUPS at the top of this module.

    Parameters
    ----------
    rmi_dfs, imv_dfs       : EUROMOD outputs by year.
    exclude_regions         : regions excluded from all computations.
    incompatible_regions    : bsarg_s zeroed in IMV run.
    region_names            : drgn2 → region name mapping.

    Returns
    -------
    pd.DataFrame with columns: drgn2, region, group, exposure_group,
    imv_mean_group, rmi_mean_group, n_hh_rmi, n_hh_imv.
    """
    all_records = []

    for group in GROUPS:
        label = group["label"]
        logger.info("Computing group exposure: %s", label)

        # Collect per-year means for this group
        rmi_year_means: dict[int, dict[int, float]] = {}
        imv_year_means: dict[int, dict[int, float]] = {}
        rmi_year_n:     dict[int, dict[int, int]]   = {}
        imv_year_n:     dict[int, dict[int, int]]   = {}

        for year in sorted(rmi_dfs.keys()):
            rmi_df = rmi_dfs[year].copy()
            imv_df = imv_dfs[year].copy()

            # Zero bsarg_s for incompatible regions in IMV run
            imv_df.loc[
                imv_df["drgn2"].isin(incompatible_regions), "bsarg_s"
            ] = 0.0
            imv_df["total_post"] = imv_df["bsa00_s"] + imv_df["bsarg_s"]

            # Filter to group households
            rmi_grp = _filter_group(rmi_df, label)
            imv_grp = _filter_group(imv_df, label)

            rmi_year_means[year] = {}
            imv_year_means[year] = {}
            rmi_year_n[year]     = {}
            imv_year_n[year]     = {}

            for drgn2 in sorted(rmi_df["drgn2"].dropna().unique()):
                if drgn2 in exclude_regions:
                    continue

                # RMI group mean
                rmi_r = rmi_grp[rmi_grp["drgn2"] == drgn2]
                rmi_rec = rmi_r[rmi_r["bsarg_s"] > 0]
                rmi_w   = rmi_rec["dwt"].sum()
                rmi_mean = (
                    (rmi_rec["bsarg_s"] * rmi_rec["dwt"]).sum() / rmi_w
                    if rmi_w > 0 else np.nan
                )
                rmi_year_means[year][drgn2] = rmi_mean
                rmi_year_n[year][drgn2]     = rmi_rec["idhh"].nunique()

                # IMV group mean
                imv_r   = imv_grp[imv_grp["drgn2"] == drgn2]
                imv_pos = imv_r[imv_r["bsa00_s"] > 0]
                imv_w   = imv_pos["dwt"].sum()
                imv_mean = (
                    (imv_pos["total_post"] * imv_pos["dwt"]).sum() / imv_w
                    if imv_w > 0 else np.nan
                )
                imv_year_means[year][drgn2] = imv_mean
                imv_year_n[year][drgn2]     = imv_pos["idhh"].nunique()

        # Average across years before differencing
        all_regions = set()
        for year_dict in rmi_year_means.values():
            all_regions.update(year_dict.keys())

        for drgn2 in sorted(all_regions):
            rmi_vals = [
                rmi_year_means[y][drgn2]
                for y in sorted(rmi_dfs.keys())
                if drgn2 in rmi_year_means[y]
                and not np.isnan(rmi_year_means[y][drgn2])
            ]
            imv_vals = [
                imv_year_means[y][drgn2]
                for y in sorted(imv_dfs.keys())
                if drgn2 in imv_year_means[y]
                and not np.isnan(imv_year_means[y][drgn2])
            ]
            rmi_n_vals = [
                rmi_year_n[y].get(drgn2, 0)
                for y in sorted(rmi_dfs.keys())
            ]
            imv_n_vals = [
                imv_year_n[y].get(drgn2, 0)
                for y in sorted(imv_dfs.keys())
            ]

            if not rmi_vals or not imv_vals:
                continue

            avg_rmi_mean = np.mean(rmi_vals)
            avg_imv_mean = np.mean(imv_vals)

            all_records.append({
                "drgn2":           int(drgn2),
                "region":          region_names.get(int(drgn2), str(drgn2)),
                "group":           label,
                "group_description": group["description"],
                "imv_mean_group":  round(avg_imv_mean, 2),
                "rmi_mean_group":  round(avg_rmi_mean, 2),
                "exposure_group":  round(avg_imv_mean - avg_rmi_mean, 2),
                "n_hh_rmi_avg":    round(np.mean(rmi_n_vals), 1),
                "n_hh_imv_avg":    round(np.mean(imv_n_vals), 1),
            })

    group_df = pd.DataFrame(all_records)

    logger.info(
        "\nGroup exposure summary:\n%s",
        group_df.pivot(
            index="region", columns="group", values="exposure_group"
        ).to_string(),
    )
    return group_df
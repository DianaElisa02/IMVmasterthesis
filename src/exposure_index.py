"""
exposure_index.py

Valid values for PRIMARY_SPEC:
    "exposure_composite_hybrid" — hybrid composite (DEFAULT)
    "exposure_exp_hybrid"       — hybrid expenditure only
    "exposure_cov_hybrid"       — hybrid coverage only
    "exposure_composite_sim"    — fully simulation-based composite
    "exposure_admin"            — purely administrative (no simulation)
"""

from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

PRIMARY_SPEC: str = "exposure_composite_hybrid"


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
    std_ = series.std(ddof=1)

    if pd.isna(std_) or std_ == 0:
        raise ValueError(f"Cannot standardise variable '{series.name}': std is zero or missing.")

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
    pooled        : output of pool_dimensions — one row per region.
    region_names  : mapping from drgn2 to region name string.

    Returns
    -------
    pd.DataFrame sorted by PRIMARY_SPEC descending.
    df.attrs contains scaling parameters for all dimensions.
    """
    result = pooled.copy()
    result["region"] = result["drgn2"].map(region_names)

    std_params: dict[str, dict[str, float]] = {}

    for spec in SPECS:
        z_cols = []

        for dim in spec["dims"]:
            if dim not in result.columns:
                raise KeyError(
                    f"Required exposure dimension '{dim}' is missing from pooled data. "
                    f"Available columns: {list(result.columns)}"
                )

            z_col = f"_z_{dim}"
            z_series, mean_, std_ = _standardise(result[dim])
            result[z_col] = z_series

            std_params[dim] = {
                "raw_mean": round(mean_, 6),
                "std": round(std_, 6),
            }
            z_cols.append(z_col)

        weights = spec["weights"]

        if len(weights) != len(z_cols):
            raise ValueError(
                f"Spec '{spec['name']}' has {len(weights)} weights but "
                f"{len(z_cols)} dimensions."
            )

        result[spec["name"]] = sum(
            w * result[z] for w, z in zip(weights, z_cols)
        ).round(4)

        valid_mask = result[spec["name"]].notna()
        ranks = pd.array([pd.NA] * len(result), dtype="Int64")

        if valid_mask.any():
            ranks[valid_mask.values] = rankdata(
                result.loc[valid_mask, spec["name"]]
            ).astype(int)

        result[f"{spec['name']}_rank"] = ranks

        logger.info(
            "Spec %-35s | dims: %s | weights: %s",
            spec["name"],
            spec["dims"],
            spec["weights"],
        )

    z_temp_cols = [c for c in result.columns if c.startswith("_z_")]
    result = result.drop(columns=z_temp_cols)

    result.attrs["std_params"] = std_params
    result.attrs["primary_spec"] = PRIMARY_SPEC

    if PRIMARY_SPEC not in result.columns:
        raise KeyError(
            f"PRIMARY_SPEC '{PRIMARY_SPEC}' was not computed. "
            f"Valid values: {[s['name'] for s in SPECS]}"
        )

    display = result.sort_values(PRIMARY_SPEC, ascending=False)
    spec_cols = [s["name"] for s in SPECS]

    logger.info(
        "\nRegional exposure index (sorted by %s):\n%s",
        PRIMARY_SPEC,
        display[["region", "drgn2"] + spec_cols].to_string(index=False),
    )

    return display.reset_index(drop=True)
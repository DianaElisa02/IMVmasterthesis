"""
exposure_index.py
=================

Constructs separate regional exposure measures for the extensive and intensive
margins of the IMV reform.

Co-primary specifications
-------------------------
1. exposure_cov_hybrid
   Change in coverage among poor households:
   simulated post-reform coverage minus administrative pre-reform RMI coverage.

2. exposure_exp_hybrid
   Change in average annual benefit among recipient households:
   simulated post-reform average benefit minus administrative pre-reform
   RMI average benefit.

The name exposure_exp_hybrid is retained for compatibility with the existing
analysis pipeline. It no longer represents expenditure per resident.

No composite exposure variable is constructed. Coverage and average benefit
are analysed separately as co-primary specifications.
"""
from __future__ import annotations

import logging

import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)


# Both hybrid margins are primary. The first is used only for default sorting.
PRIMARY_SPECS: list[str] = [
    "exposure_cov_hybrid",
    "exposure_exp_hybrid",
]

DEFAULT_SORT_SPEC: str = "exposure_cov_hybrid"


SPECS: list[dict] = [
    {
        "name": "exposure_cov_hybrid",
        "dims": ["delta_cov_hybrid"],
        "description": (
            "Hybrid coverage margin — simulated post-reform coverage among "
            "poor households minus administrative pre-reform RMI coverage"
        ),
        "primary": True,
    },
    {
        "name": "exposure_exp_hybrid",
        "dims": ["delta_benefit_hybrid"],
        "description": (
            "Hybrid average-benefit margin — simulated post-reform average "
            "annual benefit minus administrative pre-reform RMI average benefit"
        ),
        "primary": True,
    },
    {
        "name": "exposure_cov_sim",
        "dims": ["delta_cov_sim"],
        "description": (
            "Fully simulated coverage margin — simulated post-reform coverage "
            "minus simulated pre-reform RMI coverage"
        ),
        "primary": False,
    },
    {
        "name": "exposure_exp_sim",
        "dims": ["delta_benefit_sim"],
        "description": (
            "Fully simulated average-benefit margin — simulated post-reform "
            "average annual benefit minus simulated pre-reform RMI average benefit"
        ),
        "primary": False,
    },
    {
        "name": "exposure_cov_admin",
        "dims": ["level_cov_admin"],
        "description": (
            "Administrative coverage exposure — negative pre-reform RMI "
            "coverage among poor households"
        ),
        "primary": False,
    },
    {
        "name": "exposure_exp_admin",
        "dims": ["level_benefit_admin"],
        "description": (
            "Administrative average-benefit exposure — negative pre-reform "
            "RMI average annual benefit"
        ),
        "primary": False,
    },
]


def _standardise(series: pd.Series) -> tuple[pd.Series, float, float]:
    """
    Mean-centre a dimension and divide it by its cross-regional sample standard
    deviation.

    The resulting variable has cross-regional mean zero and sample standard
    deviation one. A one-unit increase therefore corresponds to a one-standard-
    deviation increase in the underlying exposure dimension.

    Returns
    -------
    standardised_series, raw_mean, raw_standard_deviation
    """
    mean_ = series.mean()
    std_ = series.std(ddof=1)

    if pd.isna(std_) or std_ == 0:
        raise ValueError(
            f"Cannot standardise variable '{series.name}': "
            "standard deviation is zero or missing."
        )

    return (series - mean_) / std_, mean_, std_


def _rank_exposure(series: pd.Series) -> pd.Series:
    """
    Rank valid exposure observations from lowest to highest.

    Missing exposure values remain missing. The 'average' method handles ties
    explicitly rather than silently truncating fractional ranks.
    """
    ranks = pd.Series(pd.NA, index=series.index, dtype="Float64")
    valid = series.notna()

    if valid.any():
        ranks.loc[valid] = rankdata(
            series.loc[valid],
            method="average",
        )

    return ranks


def compute_exposure(
    pooled: pd.DataFrame,
    region_names: dict[int, str],
) -> pd.DataFrame:
    """
    Construct separate exposure specifications from pooled regional dimensions.

    Each specification contains only one economic margin. Consequently, no
    weighting or composite-index calculation is required.

    The raw dimension is mean-centred and divided by its cross-regional sample
    standard deviation to facilitate coefficient comparisons. The
    unstandardised dimensions remain in the returned DataFrame.

    Parameters
    ----------
    pooled
        Output of pool_dimensions(), containing one row per region.

    region_names
        Mapping from regional code to regional name.

    Returns
    -------
    pd.DataFrame
        Regional exposure dataset containing separate coverage and
        average-benefit specifications and their rankings.
    """
    result = pooled.copy()
    result["region"] = result["drgn2"].map(region_names)

    scaling_params: dict[str, dict[str, float]] = {}

    for spec in SPECS:
        if len(spec["dims"]) != 1:
            raise ValueError(
                f"Specification '{spec['name']}' must contain exactly one "
                "dimension because composite exposure measures have been dropped."
            )

        dim = spec["dims"][0]

        if dim not in result.columns:
            raise KeyError(
                f"Required exposure dimension '{dim}' is missing from pooled data. "
                f"Available columns: {list(result.columns)}"
            )

        scaled, mean_, std_ = _standardise(result[dim])
        result[spec["name"]] = scaled.round(4)
        result[f"{spec['name']}_rank"] = _rank_exposure(
            result[spec["name"]]
        )

        scaling_params[dim] = {
            "raw_mean": round(float(mean_), 6),
            "std": round(float(std_), 6),
        }

        logger.info(
            "Spec %-30s | dimension: %-25s | primary: %s",
            spec["name"],
            dim,
            spec["primary"],
        )

    missing_primary = [
        spec for spec in PRIMARY_SPECS
        if spec not in result.columns
    ]
    if missing_primary:
        raise KeyError(
            f"Primary specifications were not computed: {missing_primary}. "
            f"Available specifications: {[s['name'] for s in SPECS]}"
        )

    result.attrs["std_params"] = scaling_params
    result.attrs["primary_specs"] = PRIMARY_SPECS
    result.attrs["default_sort_spec"] = DEFAULT_SORT_SPEC

    spec_cols = [spec["name"] for spec in SPECS]

    display = result.sort_values(
        DEFAULT_SORT_SPEC,
        ascending=False,
        na_position="last",
    )

    logger.info(
        "\nRegional exposure measures "
        "(sorted by %s):\n%s",
        DEFAULT_SORT_SPEC,
        display[
            ["region", "drgn2"] + spec_cols
        ].to_string(index=False),
    )

    return display.reset_index(drop=True)

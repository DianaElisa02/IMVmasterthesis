"""
exposure_dimensions.py
======================
Computes and pools raw regional gain dimensions for the IMV exposure pipeline.

Responsibilities
----------------
1. compute_regional_dimensions() — per-year, per-region raw simulated values
2. pool_dimensions()             — average before differencing, merge
                                   administrative Informe data, compute
                                   all delta dimensions

Design principle — average before differencing
-----------------------------------------------
Raw annual values are averaged across 2017, 2018, 2019 BEFORE computing
pooled deltas. Differencing noisy yearly estimates first and then averaging
compounds sampling error; averaging the raw series first reduces noise
in each component by √3 before the subtraction.

Delta dimensions produced
--------------------------
The preferred decomposition uses two standard programme margins:

1. Average annual benefit among recipient households:

    average_benefit = annual expenditure / recipient households

2. Coverage among pre-reform poor households:

    coverage = recipient households / poor households

Coverage is stored as a proportion. For example, 0.25 corresponds to
25 percent, and a difference of 0.10 corresponds to 10 percentage points.
It is NOT multiplied by 100.

Hybrid (post-reform simulation vs administrative RMI):
    delta_benefit_hybrid = simulated post-reform average annual benefit
                           - administrative RMI average annual benefit
    delta_cov_hybrid     = simulated post-reform coverage among poor households
                           - administrative RMI coverage among poor households

Fully simulated (both sides from EUROMOD):
    delta_benefit_sim    = simulated post-reform average annual benefit
                           - simulated RMI average annual benefit
    delta_cov_sim        = simulated post-reform coverage among poor households
                           - simulated RMI coverage among poor households

Purely administrative (no simulation):
    level_benefit_admin  = - administrative RMI average annual benefit
    level_cov_admin      = - administrative RMI coverage among poor households

The negative sign in the administrative-only dimensions preserves the exposure
interpretation: weaker pre-reform RMI provision means greater exposure to the
national IMV reform.

Descriptive / validation outputs:
    delta_mean           = simulated post-reform monthly mean benefit
                           - simulated RMI monthly mean benefit
    delta_exp_sim_yr     = annual simulated expenditure gap per resident,
                           retained for the pre-existing stability validation

For incompatible regions (Galicia 11, Illes Balears 53, Andalucía 61):
    bsarg_s in the IMV run is zeroed → post_protection = bsa00_s only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RECIPIENT_FLOOR: float = 10.0

def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """Return the weighted median after dropping missing/non-positive weights."""
    valid = values.notna() & weights.notna() & weights.gt(0)

    if not valid.any():
        raise ValueError("Cannot compute weighted median: no valid observations.")

    ordered = (
        pd.DataFrame({
            "value": values.loc[valid].astype(float),
            "weight": weights.loc[valid].astype(float),
        })
        .sort_values("value")
    )

    cutoff = ordered["weight"].sum() / 2
    return float(
        ordered.loc[ordered["weight"].cumsum().ge(cutoff), "value"].iloc[0]
    )

def _deduplicate_positive_benefits(
    df: pd.DataFrame,
    benefit_col: str,
    label: str,
) -> pd.DataFrame:
    """
    Keep one record per household for benefit calculations.

    This avoids over-counting rare cases where the same positive household
    benefit is repeated across more than one person row.
    """
    required = {"idhh", "dwt", benefit_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{label}: cannot deduplicate benefit records. "
            f"Missing columns: {sorted(missing)}"
        )

    dup_mask = df[df[benefit_col] > 0]["idhh"].duplicated(keep=False)
    if dup_mask.any():
        duplicated_hh = df.loc[df[benefit_col] > 0, "idhh"][dup_mask].nunique()
        logger.warning(
            "%s: %d households have repeated positive %s records. "
            "Using one record per household for recipient and expenditure calculations.",
            label,
            duplicated_hh,
            benefit_col,
        )

    return df.drop_duplicates(subset="idhh").copy()


def _prepare_poverty_denominator(rmi_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Construct the pre-reform poor-household denominator for a given ECV year.

    Poor households are defined using the standard at-risk-of-poverty concept:
    equivalised disposable income below 60 percent of the national weighted
    median. The threshold is national, not region-specific.
    """
    required = {"idhh", "drgn2", "dwt", "yds", "dag"}
    missing = required - set(rmi_df.columns)
    if missing:
        raise KeyError(
            "Cannot construct the poor-household denominator for coverage. "
            f"Missing columns: {sorted(missing)}"
        )

    hh_comp = (
        rmi_df[["idhh", "dag"]]
        .dropna()
        .assign(
            adult=lambda x: x["dag"] >= 14,
            child=lambda x: x["dag"] < 14,
        )
        .groupby("idhh")
        .agg(
            n_adults=("adult", "sum"),
            n_children=("child", "sum"),
        )
        .reset_index()
    )

    hh_comp["oecd_m_reconstructed"] = (
        1
        + 0.5 * (hh_comp["n_adults"] - 1).clip(lower=0)
        + 0.3 * hh_comp["n_children"]
    )

    hh_comp["hh_size_reconstructed"] = (
        hh_comp["n_adults"] + hh_comp["n_children"]
    )

    poverty_hh = (
        rmi_df[["idhh", "drgn2", "dwt", "yds"]]
        .drop_duplicates(subset="idhh")
        .merge(hh_comp[["idhh", "oecd_m_reconstructed", "hh_size_reconstructed"]], on="idhh", how="left")
        .copy()
    )

    poverty_hh = poverty_hh[
        poverty_hh["dwt"].notna()
        & poverty_hh["dwt"].gt(0)
        & poverty_hh["yds"].notna()
        & poverty_hh["oecd_m_reconstructed"].notna()
        & poverty_hh["oecd_m_reconstructed"].gt(0)
    ].copy()

    poverty_hh["equiv_income"] = (
        poverty_hh["yds"] / poverty_hh["oecd_m_reconstructed"]
    )

    poverty_hh["person_weight"] = poverty_hh["dwt"] * poverty_hh["hh_size_reconstructed"]

    national_median = weighted_median(
        poverty_hh["equiv_income"],
        poverty_hh["person_weight"],
    )
    poverty_threshold = 0.60 * national_median
    poverty_hh["poor_hh"] = poverty_hh["equiv_income"] < poverty_threshold

    logger.info(
        "Year %d: poverty threshold for coverage denominator = %.2f",
        year,
        poverty_threshold,
    )
    return poverty_hh

def _collapse_benefit_to_household(
    df: pd.DataFrame,
    benefit_col: str,
    label: str,
    aggregation: str,
) -> pd.DataFrame:
    """
    Collapse person-level benefit records to one household record.

    aggregation="unique_positive"
        Retains the single distinct positive amount within a household.
        Appropriate for bsa00_s, which is a household-level IMV amount.

    aggregation="sum_unique_positive"
        Sums distinct positive values within a household.
        Appropriate for bsarg_s, which can contain separate payments to
        different household members.

    Identical positive values repeated across members are counted only once.
    """
    required = {
        "idhh",
        "dwt",
        benefit_col,
    }
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            f"{label}: missing columns {sorted(missing)}"
        )

    work = df[
        ["idhh", "dwt", benefit_col]
    ].copy()

    work[benefit_col] = pd.to_numeric(
        work[benefit_col],
        errors="coerce",
    ).fillna(0.0)

    weight_counts = (
        work.groupby("idhh")["dwt"]
        .nunique(dropna=False)
    )

    inconsistent_weights = weight_counts[
        weight_counts.gt(1)
    ]

    if not inconsistent_weights.empty:
        raise ValueError(
            f"{label}: dwt differs within "
            f"{len(inconsistent_weights)} households. "
            f"Examples: {inconsistent_weights.index.tolist()[:10]}"
        )

    def aggregate_household(values: pd.Series) -> float:
        positive = (
            pd.to_numeric(values, errors="coerce")
            .dropna()
        )

        positive = positive[
            positive.gt(0)
        ].drop_duplicates()

        if positive.empty:
            return 0.0

        if aggregation == "unique_positive":
            if len(positive) > 1:
                raise ValueError(
                    f"{label}: household contains multiple distinct "
                    f"positive {benefit_col} values: "
                    f"{sorted(positive.tolist())}"
                )

            return float(positive.iloc[0])

        if aggregation == "sum_unique_positive":
            return float(positive.sum())

        raise ValueError(
            f"Unknown aggregation rule: {aggregation}"
        )

    benefits = (
        work.groupby("idhh")[benefit_col]
        .apply(aggregate_household)
        .rename(benefit_col)
        .reset_index()
    )

    weights = (
        work.groupby("idhh", as_index=False)
        .agg(dwt=("dwt", "first"))
    )

    household = weights.merge(
        benefits,
        on="idhh",
        how="left",
        validate="one_to_one",
    )

    household[benefit_col] = (
        household[benefit_col]
        .fillna(0.0)
    )

    return household

def compute_regional_dimensions(
    rmi_df: pd.DataFrame,
    imv_df: pd.DataFrame,
    year: int,
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
) -> pd.DataFrame:
    """
    Compute annual regional coverage and average-benefit dimensions.

    Benefit variables are first collapsed from person rows to household level:

    - bsa00_s uses one unique positive household amount;
    - bsarg_s sums distinct positive amounts within the household.

    The post-reform benefit is constructed only after the two components have
    been aggregated separately to household level.
    """
    imv = imv_df.copy()

    # Remove the regional RMI component where it is structurally incompatible
    # or contains the known €1 placeholder. In these regions, post-reform
    # protection is represented by the national IMV component only.
    post_bsarg_exclude_regions = (
        set(incompatible_regions)
        | {23, 24}
    )

    imv.loc[
        imv["drgn2"].isin(post_bsarg_exclude_regions),
        "bsarg_s",
    ] = 0.0

    # Do not construct total_post at person level. The two components may be
    # recorded on different household-member rows and must first be collapsed
    # separately to household level.
    poverty_hh = _prepare_poverty_denominator(
        rmi_df,
        year,
    )

    results: list[dict] = []

    regions = sorted(
        rmi_df["drgn2"]
        .dropna()
        .unique()
    )

    for drgn2 in regions:
        drgn2 = int(drgn2)

        if drgn2 in exclude_regions:
            continue

        r = rmi_df[
            rmi_df["drgn2"].eq(drgn2)
        ].copy()

        i = imv[
            imv["drgn2"].eq(drgn2)
        ].copy()

        # Retained as the legacy resident denominator. This is a person-level
        # population total because dwt is summed across person records.
        pop = r["dwt"].sum()

        if pd.isna(pop) or pop <= 0:
            logger.warning(
                "Year %d, region %d: non-positive population weight; skipping.",
                year,
                drgn2,
            )
            continue

        poor_r = poverty_hh[
            poverty_hh["drgn2"].eq(drgn2)
            & poverty_hh["poor_hh"]
        ]

        poor_hh_w = poor_r["dwt"].sum()

        if pd.isna(poor_hh_w) or poor_hh_w <= 0:
            raise ValueError(
                f"Year {year}, region {drgn2}: weighted number of poor "
                "households is zero, so coverage cannot be defined."
            )

        # ------------------------------------------------------------------
        # Pre-reform simulated RMI
        # ------------------------------------------------------------------
        # bsarg_s can contain distinct positive payments assigned to different
        # household members. Sum distinct positive values within each household
        # while avoiding duplication of identical repeated amounts.
        r_benefit = _collapse_benefit_to_household(
            r,
            "bsarg_s",
            f"RMI {year}, region {drgn2}",
            aggregation="sum_unique_positive",
        )

        # ------------------------------------------------------------------
        # Post-reform simulated protection
        # ------------------------------------------------------------------
        # bsa00_s is the national IMV household amount. The audit found no
        # households with multiple distinct positive bsa00_s values, so retain
        # the single unique positive amount.
        post_imv = _collapse_benefit_to_household(
            i,
            "bsa00_s",
            f"IMV component {year}, region {drgn2}",
            aggregation="unique_positive",
        ).rename(
            columns={
                "bsa00_s": "bsa00_hh",
            }
        )

        # bsarg_s may contain separate regional payments assigned to different
        # members, so distinct positive amounts are summed within household.
        post_rmi = _collapse_benefit_to_household(
            i,
            "bsarg_s",
            f"Regional component {year}, region {drgn2}",
            aggregation="sum_unique_positive",
        )

        # Retain the weight from post_imv and only the regional benefit from
        # post_rmi, avoiding duplicate dwt columns in the merge.
        post_rmi = post_rmi[
            ["idhh", "bsarg_s"]
        ].rename(
            columns={
                "bsarg_s": "bsarg_hh",
            }
        )

        i_benefit = post_imv.merge(
            post_rmi,
            on="idhh",
            how="outer",
            validate="one_to_one",
        )

        i_benefit["bsa00_hh"] = (
            i_benefit["bsa00_hh"]
            .fillna(0.0)
        )

        i_benefit["bsarg_hh"] = (
            i_benefit["bsarg_hh"]
            .fillna(0.0)
        )

        i_benefit["total_post"] = (
            i_benefit["bsa00_hh"]
            + i_benefit["bsarg_hh"]
        )

        # ------------------------------------------------------------------
        # Recipient classification
        # ------------------------------------------------------------------
        rmi_rec = r_benefit[
            r_benefit["bsarg_s"].ge(
                RECIPIENT_FLOOR
            )
        ].copy()

        post_rec = i_benefit[
            i_benefit["total_post"].ge(
                RECIPIENT_FLOOR
            )
        ].copy()

        rmi_rec_w = rmi_rec["dwt"].sum()
        post_rec_w = post_rec["dwt"].sum()

        # ------------------------------------------------------------------
        # Expenditure
        # ------------------------------------------------------------------
        # Numerators and recipient denominators are calculated from the same
        # eligible household records.
        rmi_exp = (
            rmi_rec["bsarg_s"]
            * rmi_rec["dwt"]
        ).sum() * 12

        post_exp = (
            post_rec["total_post"]
            * post_rec["dwt"]
        ).sum() * 12

        # ------------------------------------------------------------------
        # Monthly weighted mean among recipient households
        # ------------------------------------------------------------------
        rmi_mean = (
            (
                rmi_rec["bsarg_s"]
                * rmi_rec["dwt"]
            ).sum()
            / rmi_rec_w
            if rmi_rec_w > 0
            else np.nan
        )

        post_mean = (
            (
                post_rec["total_post"]
                * post_rec["dwt"]
            ).sum()
            / post_rec_w
            if post_rec_w > 0
            else np.nan
        )

        # ------------------------------------------------------------------
        # Annual average benefit among recipient households
        # ------------------------------------------------------------------
        rmi_avg_benefit = (
            rmi_exp / rmi_rec_w
            if rmi_rec_w > 0
            else np.nan
        )

        post_avg_benefit = (
            post_exp / post_rec_w
            if post_rec_w > 0
            else np.nan
        )

        # ------------------------------------------------------------------
        # Coverage among pre-reform poor households
        # ------------------------------------------------------------------
        rmi_coverage = (
            rmi_rec_w / poor_hh_w
        )

        post_coverage = (
            post_rec_w / poor_hh_w
        )

        results.append({
            "drgn2": drgn2,
            "year": year,
            "pop": pop,
            "poor_hh_sim": round(
                poor_hh_w,
                0,
            ),

            "rmi_exp_sim": round(
                rmi_exp,
                0,
            ),
            "imv_exp_sim": round(
                post_exp,
                0,
            ),
            "post_exp_sim": round(
                post_exp,
                0,
            ),

            "rmi_rec_sim": round(
                rmi_rec_w,
                0,
            ),
            "imv_rec_sim": round(
                post_rec_w,
                0,
            ),
            "post_rec_sim": round(
                post_rec_w,
                0,
            ),

            "rmi_mean_sim": (
                round(rmi_mean, 2)
                if pd.notna(rmi_mean)
                else np.nan
            ),
            "imv_mean_sim": (
                round(post_mean, 2)
                if pd.notna(post_mean)
                else np.nan
            ),

            "rmi_avg_benefit_sim": (
                round(rmi_avg_benefit, 2)
                if pd.notna(rmi_avg_benefit)
                else np.nan
            ),
            "post_avg_benefit_sim": (
                round(post_avg_benefit, 2)
                if pd.notna(post_avg_benefit)
                else np.nan
            ),

            "rmi_coverage_sim": round(
                rmi_coverage,
                6,
            ),
            "post_coverage_sim": round(
                post_coverage,
                6,
            ),

            "delta_benefit_sim_yr": (
                round(
                    post_avg_benefit
                    - rmi_avg_benefit,
                    4,
                )
                if (
                    pd.notna(post_avg_benefit)
                    and pd.notna(rmi_avg_benefit)
                )
                else np.nan
            ),

            "delta_cov_sim_yr": round(
                post_coverage
                - rmi_coverage,
                6,
            ),

            # Retained for legacy descriptive or validation use.
            "delta_exp_sim_yr": (
                round(
                    (
                        post_exp
                        - rmi_exp
                    )
                    / pop,
                    4,
                )
                if pop > 0
                else np.nan
            ),
        })

    df = pd.DataFrame(results)

    logger.info(
        "Year %d: computed raw dimensions for %d regions",
        year,
        len(df),
    )

    return df

def pool_dimensions(
    rmi_dfs: dict[int, pd.DataFrame],
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    incompatible_regions: frozenset[int],
    informe_rmi: dict[int, list[dict]],
    region_population: dict[int, dict[int, int]],
    sim_exclude_regions: frozenset[int] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    sim_exclude_regions : regions to exclude ONLY from fully simulated deltas.
        Hybrid and admin deltas are still computed for these regions.

        Use case: La Rioja (23) and Aragón (24) have a broken bsarg_s in
        the pre-reform STD files (€1 placeholder), so their simulated RMI
        expenditure and recipient counts are unreliable. The hybrid specs are
        clean because the RMI side uses Informe administrative data.
    """
    frames = []
    for year in sorted(rmi_dfs.keys()):
        frames.append(
            compute_regional_dimensions(
                rmi_dfs[year], imv_dfs[year], year,
                exclude_regions, incompatible_regions,
            )
        )
    all_dims = pd.concat(frames, ignore_index=True)

    pooled = (
        all_dims.groupby("drgn2")
        .agg(
            pop                   = ("pop",                   "mean"),
            poor_hh_sim           = ("poor_hh_sim",           "mean"),

            rmi_exp_sim           = ("rmi_exp_sim",           "mean"),
            imv_exp_sim           = ("imv_exp_sim",           "mean"),
            post_exp_sim          = ("post_exp_sim",          "mean"),

            rmi_rec_sim           = ("rmi_rec_sim",           "mean"),
            imv_rec_sim           = ("imv_rec_sim",           "mean"),
            post_rec_sim          = ("post_rec_sim",          "mean"),

            rmi_mean_sim          = ("rmi_mean_sim",          "mean"),
            imv_mean_sim          = ("imv_mean_sim",          "mean"),
            rmi_avg_benefit_sim   = ("rmi_avg_benefit_sim",   "mean"),
            post_avg_benefit_sim  = ("post_avg_benefit_sim",  "mean"),
            rmi_coverage_sim      = ("rmi_coverage_sim",      "mean"),
            post_coverage_sim     = ("post_coverage_sim",     "mean"),
        )
        .reset_index()
        .round(6)
    )

    admin_records = []
    years = sorted(informe_rmi.keys())
    for year in years:
        pop_year = region_population.get(year, {})
        for row in informe_rmi[year]:
            drgn2 = row["drgn2"]
            if drgn2 in exclude_regions:
                continue
            admin_records.append({
                "drgn2":         drgn2,
                "year":          year,
                "rmi_exp_admin": row["gasto_anual_ejecutado"],
                "titulares":     row["titulares"],
                "pop_admin":     pop_year.get(drgn2, np.nan),
            })

    admin_df = (
        pd.DataFrame(admin_records)
        .groupby("drgn2")
        .agg(
            avg_rmi_exp_admin   = ("rmi_exp_admin", "mean"),
            avg_titulares_admin = ("titulares",      "mean"),
            avg_pop_admin       = ("pop_admin",      "mean"),
        )
        .reset_index()
        .round(2)
    )

    pooled = pooled.merge(admin_df, on="drgn2", how="left")

    pooled["rmi_avg_benefit_admin"] = np.where(
        pooled["avg_titulares_admin"].gt(0),
        pooled["avg_rmi_exp_admin"] / pooled["avg_titulares_admin"],
        np.nan,
    )
    pooled["rmi_coverage_admin"] = (
        pooled["avg_titulares_admin"] / pooled["poor_hh_sim"]
    )

    # Hybrid dimensions: simulated post-reform side, administrative RMI side.
    pooled["delta_benefit_hybrid"] = (
        pooled["post_avg_benefit_sim"] - pooled["rmi_avg_benefit_admin"]
    ).round(4)
    pooled["delta_cov_hybrid"] = (
        pooled["post_coverage_sim"] - pooled["rmi_coverage_admin"]
    ).round(6)

    # Fully simulated dimensions: both sides from EUROMOD.
    pooled["delta_benefit_sim"] = (
        pooled["post_avg_benefit_sim"] - pooled["rmi_avg_benefit_sim"]
    ).round(4)
    pooled["delta_cov_sim"] = (
        pooled["post_coverage_sim"] - pooled["rmi_coverage_sim"]
    ).round(6)

    # Regions with broken simulated RMI values should not enter the fully
    # simulated robustness exposure. Do not substitute admin data and then label
    # the result fully simulated.
    if sim_exclude_regions:
        sim_mask = pooled["drgn2"].isin(sim_exclude_regions)
        pooled.loc[sim_mask, ["delta_benefit_sim", "delta_cov_sim"]] = np.nan
        logger.info(
            "Set fully simulated benefit/coverage deltas to missing for regions "
            "with broken EUROMOD RMI placeholder values: %s",
            sorted(sim_exclude_regions),
        )

    # Administrative-only dimensions: lower pre-reform provision = higher exposure.
    pooled["level_benefit_admin"] = (-pooled["rmi_avg_benefit_admin"]).round(4)
    pooled["level_cov_admin"] = (-pooled["rmi_coverage_admin"]).round(6)

    # Backward-compatible aliases for older scripts/output names. These now refer
    # to average-benefit and poor-household-coverage dimensions, not expenditure
    # per resident or recipients per population.
    pooled["delta_exp_hybrid"] = pooled["delta_benefit_hybrid"]
    pooled["delta_exp_sim"] = pooled["delta_benefit_sim"]
    pooled["delta_exp_admin"] = pooled["level_benefit_admin"]
    pooled["delta_cov_admin"] = pooled["level_cov_admin"]

    pooled["delta_mean"] = (
        pooled["imv_mean_sim"] - pooled["rmi_mean_sim"]
    ).round(2)

    if sim_exclude_regions:
        all_dims.loc[
            all_dims["drgn2"].isin(sim_exclude_regions),
            ["delta_exp_sim_yr", "delta_benefit_sim_yr", "delta_cov_sim_yr"],
        ] = np.nan

    logger.info(
        "Pooled %d years → %d regions (average before differencing)",
        len(rmi_dfs), len(pooled),
    )
    logger.info(
        "\nPooled dimensions:\n%s",
        pooled[[
            "drgn2",
            "delta_benefit_hybrid", "delta_cov_hybrid",
            "delta_benefit_sim",    "delta_cov_sim",
            "level_benefit_admin",  "level_cov_admin",
            "delta_mean",
        ]].to_string(index=False),
    )
    return pooled, all_dims

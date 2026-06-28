"""
exposure_validation.py
======================

Statistical validation of:

A. The EUROMOD IMV simulation.
B. The separate regional exposure measures:
   - coverage among pre-reform poor households;
   - average annual benefit among recipient households.

The former composite exposure index is no longer constructed or validated.

Validation tasks
----------------

A — IMV simulation quality

Test 1 — Benefit bounds
    Checks monthly bsa00_s among household-level recipients. A recipient is
    defined consistently with exposure construction as receiving at least the
    statutory minimum monthly payment floor.

Test 3 — Income means test
    Checks that recipient households have lower disposable household income
    than non-recipient households.

Test 6 — Formula plausibility
    Checks whether single-person recipient households receive benefits broadly
    consistent with the statutory single-adult guaranteed-income amount.

B — Exposure-measure diagnostics

Test 4 — Annual dimension stability
    Checks the cross-regional rank stability of:
      - delta_benefit_sim_yr;
      - delta_cov_sim_yr.

Test 5 — IMV regional rank stability
    Checks the stability of regional average IMV benefits across ECV waves.
    This is treated as an informative diagnostic rather than a pass/fail test
    because the IMV rules are national and regional variation reflects sampled
    household composition.

Test 7 — Coverage-margin institutional consistency
    Checks whether hybrid coverage exposure is negatively associated with
    administrative pre-reform RMI coverage among poor households.

Test 8 — Average-benefit-margin institutional consistency
    Checks whether hybrid average-benefit exposure is negatively associated
    with administrative pre-reform average annual RMI benefits.

The institutional-consistency tests are directional diagnostics. Because the
administrative baseline enters the hybrid difference directly, they should not
be interpreted as independent external validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

logger = logging.getLogger(__name__)


def _household_level(
    df: pd.DataFrame,
    required_columns: list[str],
) -> pd.DataFrame:
    """
    Return one row per household.

    bsa00_s is aggregated using the unique positive household amount. The
    EUROMOD output may record the positive amount on one household-member row
    and zero on the others, so arbitrary first-row deduplication is unsafe.

    Other household-level variables are retained from the first row after
    checking that they are constant within household.
    """
    required = {"idhh", *required_columns}
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            "Cannot construct household-level validation data. "
            f"Missing columns: {sorted(missing)}"
        )

    work = df[["idhh", *required_columns]].copy()

    if "bsa00_s" in work.columns:
        work["bsa00_s"] = pd.to_numeric(
            work["bsa00_s"],
            errors="coerce",
        ).fillna(0.0)

    non_benefit_columns = [
        column
        for column in required_columns
        if column != "bsa00_s"
    ]

    # Verify that variables treated as household-level are constant within idhh.
    for column in non_benefit_columns:
        counts = (
            work.groupby("idhh")[column]
            .nunique(dropna=False)
        )

        inconsistent = counts[counts.gt(1)]

        if not inconsistent.empty:
            raise ValueError(
                f"Variable '{column}' differs within "
                f"{len(inconsistent)} households. Examples: "
                f"{inconsistent.index.tolist()[:10]}"
            )

    household = (
        work.groupby("idhh", as_index=False)
        .agg(
            **{
                column: (column, "first")
                for column in non_benefit_columns
            }
        )
    )

    if "bsa00_s" in work.columns:
        positive_counts = (
            work.loc[work["bsa00_s"].gt(0)]
            .groupby("idhh")["bsa00_s"]
            .nunique()
        )

        conflicting = positive_counts[
            positive_counts.gt(1)
        ]

        if not conflicting.empty:
            raise ValueError(
                f"{len(conflicting)} households contain multiple distinct "
                "positive bsa00_s values. Examples: "
                f"{conflicting.index.tolist()[:10]}"
            )

        bsa00_household = (
            work.groupby("idhh", as_index=False)
            .agg(
                bsa00_s=("bsa00_s", "max"),
            )
        )

        household = household.merge(
            bsa00_household,
            on="idhh",
            how="left",
            validate="one_to_one",
        )

    return household

def _safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float, int]:
    """
    Compute Spearman correlation using complete observations.

    Returns NaN values when fewer than three usable observations remain or
    either variable has no cross-sectional variation.
    """
    valid = x.notna() & y.notna()
    x_valid = x.loc[valid]
    y_valid = y.loc[valid]
    n = int(valid.sum())

    if n < 3 or x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return np.nan, np.nan, n

    rho, pval = spearmanr(x_valid, y_valid)
    return float(rho), float(pval), n


# =============================================================================
# A. IMV simulation-quality tests
# =============================================================================

def test_benefit_bounds(
    df: pd.DataFrame,
    year: int,
    statutory_min: float,
    statutory_max: float,
    floor_monthly: float = 10.0,
) -> dict:
    """
    Test 1: household-level IMV benefits should respect plausible bounds.

    Recipient status is defined consistently with the exposure construction:
    bsa00_s >= floor_monthly.

    The upper bound allows a 10 percent tolerance for supplement combinations.
    """
    hh = _household_level(
        df,
        ["bsa00_s", "dwt"],
    )

    recipients = hh[
        hh["bsa00_s"].notna()
        & hh["dwt"].notna()
        & hh["dwt"].gt(0)
        & hh["bsa00_s"].ge(floor_monthly)
    ].copy()

    if recipients.empty:
        result = {
            "test": "benefit_bounds",
            "year": year,
            "pass": None,
            "note": "no household recipients at or above the payment floor",
        }
        logger.warning(
            "[INFO] Test 1 — Benefit bounds %d: no eligible recipients",
            year,
        )
        return result

    n = len(recipients)
    weighted_n = recipients["dwt"].sum()

    below_floor = int(
        recipients["bsa00_s"].lt(floor_monthly).sum()
    )
    above_max = int(
        recipients["bsa00_s"].gt(statutory_max * 1.10).sum()
    )

    weighted_mean = (
        recipients["bsa00_s"] * recipients["dwt"]
    ).sum() / weighted_n

    result = {
        "test": "benefit_bounds",
        "year": year,
        "n_recipient_households_unweighted": n,
        "n_recipient_households_weighted": round(weighted_n, 0),
        "mean_monthly_benefit": round(weighted_mean, 2),
        "statutory_single_GMI": statutory_min,
        "statutory_max_GMI": statutory_max,
        "payment_floor": floor_monthly,
        "n_below_floor": below_floor,
        "n_above_max_110pct": above_max,
        "pass": below_floor == 0 and above_max == 0,
    }

    status = "PASS" if result["pass"] else "WARN"

    logger.info(
        "[%s] Test 1 — Benefit bounds %d: "
        "household mean=€%.2f, below_floor=%d, above_max(110%%)=%d",
        status,
        year,
        weighted_mean,
        below_floor,
        above_max,
    )

    return result


def test_income_means_test(
    df: pd.DataFrame,
    year: int,
    floor_monthly: float = 10.0,
) -> dict:
    """
    Test 3: recipient households should have lower disposable household income
    than non-recipient households.

    The comparison is performed at household level. Households receiving less
    than the statutory payment floor are classified as non-recipients.
    """
    hh = _household_level(
        df,
        ["bsa00_s", "yds"],
    )

    valid = hh["bsa00_s"].notna() & hh["yds"].notna()
    hh = hh.loc[valid].copy()

    recipients = hh.loc[
        hh["bsa00_s"].ge(floor_monthly),
        "yds",
    ]

    non_recipients = hh.loc[
        hh["bsa00_s"].lt(floor_monthly),
        "yds",
    ]

    if recipients.empty or non_recipients.empty:
        result = {
            "test": "income_means_test",
            "year": year,
            "pass": None,
            "note": "recipient or non-recipient comparison group is empty",
        }
        logger.warning(
            "[INFO] Test 3 — Income means test %d: comparison unavailable",
            year,
        )
        return result

    statistic, pval = mannwhitneyu(
        recipients,
        non_recipients,
        alternative="less",
    )

    mean_recipient_income = recipients.mean()
    mean_nonrecipient_income = non_recipients.mean()

    result = {
        "test": "income_means_test",
        "year": year,
        "n_hh_recipients": len(recipients),
        "n_hh_non_recipients": len(non_recipients),
        "mean_yds_recipients": round(mean_recipient_income, 2),
        "mean_yds_non_recipients": round(mean_nonrecipient_income, 2),
        "income_ratio": round(
            mean_recipient_income / mean_nonrecipient_income,
            3,
        ),
        "mannwhitney_stat": round(statistic, 0),
        "mannwhitney_p": round(pval, 6),
        "pass": (
            pval < 0.05
            and mean_recipient_income < mean_nonrecipient_income
        ),
    }

    status = "PASS" if result["pass"] else "WARN"

    logger.info(
        "[%s] Test 3 — Income means test %d: "
        "mean_yds_rec=€%.0f vs non_rec=€%.0f (p=%.4f) "
        "[N_hh: %d recipients, %d non-recipients]",
        status,
        year,
        mean_recipient_income,
        mean_nonrecipient_income,
        pval,
        len(recipients),
        len(non_recipients),
    )

    return result


def test_formula_plausibility(
    df: pd.DataFrame,
    year: int,
    statutory_single: float,
    floor_monthly: float = 10.0,
) -> dict:
    """
    Test 6: single-person recipient households should receive benefits broadly
    consistent with the statutory single-adult guaranteed-income amount.

    A 20 percent tolerance is retained because bsa00_s is an income top-up,
    not necessarily the full statutory guaranteed-income threshold.
    """
    required = {
        "idhh",
        "idperson",
        "bsa00_s",
        "dwt",
    }
    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            "Cannot run formula-plausibility test. "
            f"Missing columns: {sorted(missing)}"
        )

    household_size = (
        df.groupby("idhh")["idperson"]
        .nunique()
        .rename("hh_size_proxy")
    )

    hh = _household_level(
        df,
        ["bsa00_s", "dwt"],
    ).merge(
        household_size,
        left_on="idhh",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    single_recipients = hh[
        hh["bsa00_s"].ge(floor_monthly)
        & hh["dwt"].notna()
        & hh["dwt"].gt(0)
        & hh["hh_size_proxy"].eq(1)
    ].copy()

    if single_recipients.empty:
        result = {
            "test": "formula_plausibility",
            "year": year,
            "pass": None,
            "note": "no single-person recipient households found",
        }

        logger.info(
            "[INFO] Test 6 — Formula plausibility %d: "
            "no single-person recipients",
            year,
        )

        return result

    weighted_mean = (
        single_recipients["bsa00_s"]
        * single_recipients["dwt"]
    ).sum() / single_recipients["dwt"].sum()

    percentage_difference = (
        abs(weighted_mean - statutory_single)
        / statutory_single
    )

    result = {
        "test": "formula_plausibility",
        "year": year,
        "mean_bsa00_s_single_hh": round(weighted_mean, 2),
        "statutory_single_GMI": statutory_single,
        "pct_difference": round(100 * percentage_difference, 1),
        "n_single_hh_recipients": len(single_recipients),
        "pass": percentage_difference <= 0.20,
        "note": (
            "20 percent tolerance around statutory single-person "
            "guaranteed-income amount"
        ),
    }

    status = "PASS" if result["pass"] else "WARN"

    logger.info(
        "[%s] Test 6 — Formula plausibility %d: "
        "mean_single=€%.2f vs statutory=€%.2f (diff=%.1f%%)",
        status,
        year,
        weighted_mean,
        statutory_single,
        100 * percentage_difference,
    )

    return result


# =============================================================================
# B. Exposure-measure diagnostics
# =============================================================================

def test_exposure_dimension_stability(
    all_dims: pd.DataFrame,
    exclude_regions: frozenset[int],
) -> list[dict]:
    """
    Test 4: assess year-to-year regional rank stability separately for:

      - delta_benefit_sim_yr;
      - delta_cov_sim_yr.

    The old expenditure-per-resident variable is not used.

    A correlation above 0.70 with p < 0.05 is treated as a strong stability
    result. Lower correlations are reported as warnings rather than evidence
    that the exposure construction is necessarily invalid.
    """
    required = {
        "drgn2",
        "year",
        "delta_benefit_sim_yr",
        "delta_cov_sim_yr",
    }
    missing = required - set(all_dims.columns)

    if missing:
        raise KeyError(
            "Cannot test annual exposure-dimension stability. "
            f"Missing columns: {sorted(missing)}"
        )

    dims = all_dims[
        ~all_dims["drgn2"].isin(exclude_regions)
    ].copy()

    years = sorted(dims["year"].dropna().unique())
    results: list[dict] = []

    dimension_labels = {
        "delta_benefit_sim_yr": "average-benefit margin",
        "delta_cov_sim_yr": "coverage margin",
    }

    for column, label in dimension_labels.items():
        for index in range(len(years) - 1):
            year_1 = years[index]
            year_2 = years[index + 1]

            first = (
                dims[dims["year"].eq(year_1)]
                .set_index("drgn2")[column]
            )
            second = (
                dims[dims["year"].eq(year_2)]
                .set_index("drgn2")[column]
            )

            common = first.index.intersection(second.index)

            rho, pval, n_regions = _safe_spearman(
                first.loc[common],
                second.loc[common],
            )

            passed = (
                not np.isnan(rho)
                and not np.isnan(pval)
                and rho > 0.70
                and pval < 0.05
            )

            result = {
                "test": "exposure_dimension_stability_sim",
                "dimension": column,
                "dimension_label": label,
                "years": f"{year_1}_vs_{year_2}",
                "spearman_rho": (
                    round(rho, 3)
                    if not np.isnan(rho)
                    else None
                ),
                "spearman_p": (
                    round(pval, 4)
                    if not np.isnan(pval)
                    else None
                ),
                "n_regions": n_regions,
                "pass": passed,
                "note": (
                    "year-to-year rank stability of the fully simulated "
                    f"{label}"
                ),
            }

            status = "PASS" if passed else "WARN"

            logger.info(
                "[%s] Test 4 — Stability of %s, %d vs %d: "
                "rho=%s (p=%s), N=%d",
                status,
                label,
                year_1,
                year_2,
                f"{rho:.3f}" if not np.isnan(rho) else "NA",
                f"{pval:.4f}" if not np.isnan(pval) else "NA",
                n_regions,
            )

            results.append(result)

    return results


def test_regional_rank_consistency(
    imv_dfs: dict[int, pd.DataFrame],
    exclude_regions: frozenset[int],
    floor_monthly: float = 10.0,
) -> list[dict]:
    """
    Test 5: assess regional rank stability of average simulated IMV benefits.

    Household-level data and the same minimum-payment threshold used in the
    exposure construction are applied.

    This remains an informative diagnostic and does not determine whether the
    validation suite passes.
    """
    def regional_means(df: pd.DataFrame) -> pd.Series:
        required = {
            "idhh",
            "drgn2",
            "bsa00_s",
            "dwt",
        }
        missing = required - set(df.columns)

        if missing:
            raise KeyError(
                "Cannot compute regional IMV benefit means. "
                f"Missing columns: {sorted(missing)}"
        )

        hh = _household_level(
            df,
            ["drgn2", "bsa00_s", "dwt"],
        )

        recipients = hh[
            hh["bsa00_s"].ge(floor_monthly)
            & hh["dwt"].notna()
            & hh["dwt"].gt(0)
            & ~hh["drgn2"].isin(exclude_regions)
        ].copy()

        if recipients.empty:
            return pd.Series(dtype=float)

        recipients["weighted_benefit"] = (
            recipients["bsa00_s"]
            * recipients["dwt"]
        )

        regional = recipients.groupby("drgn2").agg(
            weighted_benefit_sum=(
                "weighted_benefit",
                "sum",
            ),
            weight_sum=(
                "dwt",
                "sum",
            ),
        )

        return (
            regional["weighted_benefit_sum"]
            / regional["weight_sum"]
        )

    years = sorted(imv_dfs.keys())
    results: list[dict] = []

    for index in range(len(years) - 1):
        year_1 = years[index]
        year_2 = years[index + 1]

        first = regional_means(imv_dfs[year_1])
        second = regional_means(imv_dfs[year_2])

        common = first.index.intersection(second.index)

        rho, pval, n_regions = _safe_spearman(
            first.loc[common],
            second.loc[common],
        )

        result = {
            "test": "regional_rank_consistency",
            "years": f"{year_1}_vs_{year_2}",
            "spearman_rho": (
                round(rho, 3)
                if not np.isnan(rho)
                else None
            ),
            "spearman_p": (
                round(pval, 4)
                if not np.isnan(pval)
                else None
            ),
            "n_regions": n_regions,
            "pass": None,
            "note": (
                "informative diagnostic only; the national IMV rules do not "
                "imply stable regional benefit rankings across ECV samples"
            ),
        }

        logger.info(
            "[INFO] Test 5 — IMV regional average-benefit rank "
            "%d vs %d: rho=%s (p=%s), N=%d",
            year_1,
            year_2,
            f"{rho:.3f}" if not np.isnan(rho) else "NA",
            f"{pval:.4f}" if not np.isnan(pval) else "NA",
            n_regions,
        )

        results.append(result)

    return results


def test_institutional_consistency(
    exposure_df: pd.DataFrame,
    informe_rmi: dict[int, list[dict]],
    region_population: dict[int, dict[int, int]],
    exclude_regions: frozenset[int],
) -> list[dict]:
    """
    Tests 7 and 8: directional consistency of the two co-primary hybrid
    exposure measures with their corresponding administrative baselines.

    Test 7:
        Spearman(
            exposure_cov_hybrid,
            rmi_coverage_admin
        )
        Expected sign: negative.

    Test 8:
        Spearman(
            exposure_exp_hybrid,
            rmi_avg_benefit_admin
        )
        Expected sign: negative.

    informe_rmi and region_population remain in the function signature for
    compatibility with the existing pipeline. The required administrative
    baseline variables are already contained in exposure_df after pooling.
    """
    del informe_rmi
    del region_population

    required = {
        "drgn2",
        "exposure_cov_hybrid",
        "exposure_exp_hybrid",
        "rmi_coverage_admin",
        "rmi_avg_benefit_admin",
    }
    missing = required - set(exposure_df.columns)

    if missing:
        raise KeyError(
            "Cannot run institutional-consistency diagnostics. "
            f"Missing columns: {sorted(missing)}"
        )

    data = exposure_df[
        ~exposure_df["drgn2"].isin(exclude_regions)
    ].copy()

    diagnostics = [
        {
            "test_number": 7,
            "test_name": "institutional_consistency_coverage",
            "exposure_column": "exposure_cov_hybrid",
            "baseline_column": "rmi_coverage_admin",
            "label": "coverage among pre-reform poor households",
            "note": (
                "negative association expected: stronger pre-reform RMI "
                "coverage implies a smaller post-minus-pre coverage gain"
            ),
        },
        {
            "test_number": 8,
            "test_name": "institutional_consistency_average_benefit",
            "exposure_column": "exposure_exp_hybrid",
            "baseline_column": "rmi_avg_benefit_admin",
            "label": "average annual benefit among RMI recipient households",
            "note": (
                "negative association expected: higher pre-reform average "
                "benefits imply a smaller post-minus-pre benefit gain"
            ),
        },
    ]

    results: list[dict] = []

    for diagnostic in diagnostics:
        exposure_column = diagnostic["exposure_column"]
        baseline_column = diagnostic["baseline_column"]

        rho, pval, n_regions = _safe_spearman(
            data[baseline_column],
            data[exposure_column],
        )

        passed = (
            not np.isnan(rho)
            and not np.isnan(pval)
            and rho < 0
            and pval < 0.10
        )

        result = {
            "test": diagnostic["test_name"],
            "test_number": diagnostic["test_number"],
            "exposure_spec": exposure_column,
            "admin_benchmark": baseline_column,
            "benchmark_label": diagnostic["label"],
            "n_regions": n_regions,
            "spearman_rho": (
                round(rho, 3)
                if not np.isnan(rho)
                else None
            ),
            "spearman_p": (
                round(pval, 4)
                if not np.isnan(pval)
                else None
            ),
            "pass": passed,
            "note": (
                diagnostic["note"]
                + "; directional diagnostic, not independent validation, "
                  "because the baseline enters the hybrid difference"
            ),
        }

        status = "PASS" if passed else "WARN"

        logger.info(
            "[%s] Test %d — Institutional consistency, %s: "
            "%s vs %s, rho=%s (p=%s), N=%d",
            status,
            diagnostic["test_number"],
            diagnostic["label"],
            exposure_column,
            baseline_column,
            f"{rho:.3f}" if not np.isnan(rho) else "NA",
            f"{pval:.4f}" if not np.isnan(pval) else "NA",
            n_regions,
        )

        results.append(result)

    return results


def test_coverage_levels(
    exposure_df: pd.DataFrame,
    exclude_regions: frozenset[int],
) -> list[dict]:
    """
    Additional diagnostic: report whether any coverage levels are negative or
    exceed one.

    Coverage above one is not automatically impossible because administrative
    programme recipients and income-poor households are not perfectly nested.
    Such cases are flagged for inspection rather than treated automatically as
    validation failures.
    """
    coverage_columns = [
        "rmi_coverage_sim",
        "post_coverage_sim",
        "rmi_coverage_admin",
    ]

    available = [
        column
        for column in coverage_columns
        if column in exposure_df.columns
    ]

    data = exposure_df[
        ~exposure_df["drgn2"].isin(exclude_regions)
    ].copy()

    results: list[dict] = []

    for column in available:
        values = data[column].dropna()

        if values.empty:
            continue

        n_negative = int(values.lt(0).sum())
        n_above_one = int(values.gt(1).sum())

        result = {
            "test": "coverage_level_diagnostic",
            "dimension": column,
            "n_regions": len(values),
            "minimum": round(float(values.min()), 6),
            "maximum": round(float(values.max()), 6),
            "n_negative": n_negative,
            "n_above_one": n_above_one,
            "pass": n_negative == 0,
            "note": (
                "values above one are flagged for inspection but are not "
                "automatically invalid because beneficiary and poverty "
                "populations may not be perfectly nested"
            ),
        }

        status = "PASS" if result["pass"] else "WARN"

        logger.info(
            "[%s] Coverage-level diagnostic — %s: "
            "min=%.3f, max=%.3f, below_zero=%d, above_one=%d",
            status,
            column,
            values.min(),
            values.max(),
            n_negative,
            n_above_one,
        )

        results.append(result)

    return results


# =============================================================================
# Full validation suite
# =============================================================================

def run_validation(
    imv_dfs: dict[int, pd.DataFrame],
    all_dims: pd.DataFrame,
    exposure_df: pd.DataFrame,
    informe_rmi: dict[int, list[dict]],
    region_population: dict[int, dict[int, int]],
    statutory_single: float,
    statutory_max: float,
    floor_monthly: float,
    exclude_regions: frozenset[int],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run the complete validation suite and save the results.

    The function signature is retained so compute_exposure_variable.py does
    not need to change.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    # -------------------------------------------------------------------------
    # A. IMV simulation quality
    # -------------------------------------------------------------------------
    for year, df in sorted(imv_dfs.items()):
        logger.info("=" * 60)
        logger.info("Validating IMV simulation — year %d", year)

        all_results.append(
            test_benefit_bounds(
                df=df,
                year=year,
                statutory_min=statutory_single,
                statutory_max=statutory_max,
                floor_monthly=floor_monthly,
            )
        )

        all_results.append(
            test_income_means_test(
                df=df,
                year=year,
                floor_monthly=floor_monthly,
            )
        )

        all_results.append(
            test_formula_plausibility(
                df=df,
                year=year,
                statutory_single=statutory_single,
                floor_monthly=floor_monthly,
            )
        )

    # -------------------------------------------------------------------------
    # B. Exposure-measure diagnostics
    # -------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Validating separate regional exposure measures")

    all_results.extend(
        test_exposure_dimension_stability(
            all_dims=all_dims,
            exclude_regions=exclude_regions,
        )
    )

    all_results.extend(
        test_regional_rank_consistency(
            imv_dfs=imv_dfs,
            exclude_regions=exclude_regions,
            floor_monthly=floor_monthly,
        )
    )

    all_results.extend(
        test_institutional_consistency(
            exposure_df=exposure_df,
            informe_rmi=informe_rmi,
            region_population=region_population,
            exclude_regions=exclude_regions,
        )
    )

    all_results.extend(
        test_coverage_levels(
            exposure_df=exposure_df,
            exclude_regions=exclude_regions,
        )
    )

    flat_results = [
        {
            key: value
            for key, value in result.items()
            if not isinstance(value, (list, dict))
        }
        for result in all_results
    ]

    results_df = pd.DataFrame(flat_results)

    output_path = output_dir / "imv_validation_report.csv"
    results_df.to_csv(output_path, index=False)

    logger.info(
        "Validation report saved → %s",
        output_path,
    )

    evaluated = results_df["pass"].notna()
    n_evaluated = int(evaluated.sum())
    n_passed = int(
        results_df.loc[evaluated, "pass"]
        .astype(bool)
        .sum()
    )

    logger.info(
        "Validation summary: %d/%d evaluated tests passed "
        "(informative diagnostics excluded)",
        n_passed,
        n_evaluated,
    )

    return results_df


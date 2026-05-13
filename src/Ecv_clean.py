"""
ecv_clean.py
============
Household-level ECV cleaning for the IMV DiD analysis pipeline.

Mirrors the structure of household.py and person.py:
  - Pure Polars throughout (no pandas after the read step in readers.py)
  - No static values declared here — all constants imported from src.constants
  - No I/O — reading is done in readers.py, writing in the orchestrator

Public API
----------
  build_household_analysis(td, th, tr, tp, year) -> pl.DataFrame
      Returns one row per household with all DiD variables.

  build_analysis_panel(years, ecv_dir, exposure_path) -> pl.DataFrame
      Loops over years, concatenates, drops 2020, excludes regions,
      merges exposure, constructs interactions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.constants import (
    ANALYSIS_EXCLUDE_DRGN2,
    ANALYSIS_N_CLUSTERS,
    ANALYSIS_TH_COLUMNS,
    ANALYSIS_TR_COLUMNS,
    ANALYSIS_TP_COLUMNS,
    DRGN2_MAP,
    EXPOSURE_SPECS,
    ISCED_GROUP_BOUNDARIES,
    PL031_TO_LABOUR_GROUP,
    REGION_NAMES,
)
from src.readers import _read_section

logger = logging.getLogger(__name__)


# =============================================================================
# PATH BUILDER
# Mirrors make_paths() from the old cleaning script.
# Structure: base / datos_{year} / ECV_T{x}_{year} / STATA / ECV_T{x}_{year}.dta
# =============================================================================

def _make_path(base: Path, file_type: str, year: int) -> Path:
    """
    Build the full path to an ECV STATA file for a given year.

    file_type : one of "td", "th", "tr", "tp"
    """
    prefix = f"ECV_T{file_type[1].upper()}_{year}"   # e.g. ECV_Td_2021
    return base / f"datos_{year}" / prefix / "STATA" / f"{prefix}.dta"


# =============================================================================
# READERS — use local path builder, reuse _read_section from readers.py
# =============================================================================

def read_td_analysis(base: Path, year: int) -> pl.DataFrame:
    from src.constants import TD_COLUMNS
    return _read_section(_make_path(base, "td", year), TD_COLUMNS, year, "td")


def read_th_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "th", year), ANALYSIS_TH_COLUMNS, year, "th_analysis")


def read_tr_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "tr", year), ANALYSIS_TR_COLUMNS, year, "tr_analysis")


def read_tp_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "tp", year), ANALYSIS_TP_COLUMNS, year, "tp_analysis")


# =============================================================================
# RECODE HELPERS (Polars expressions — no side effects)
# =============================================================================

def _recode_binary_outcome(col: pl.Expr) -> pl.Expr:
    """Map 1→1.0, 0→0.0, anything else→null. Used for vhMATDEP and vhPobreza."""
    num = col.cast(pl.Float64, strict=False)
    return (
        pl.when(num.eq(1.0)).then(pl.lit(1.0, dtype=pl.Float64))
        .when(num.eq(0.0)).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _recode_homeowner(hh021: pl.Expr) -> pl.Expr:
    """1 if owner (tenure 1 or 2), 0 if tenant/other, null if missing."""
    num = hh021.cast(pl.Float64, strict=False)
    return (
        pl.when(num.is_in([1.0, 2.0])).then(pl.lit(1.0, dtype=pl.Float64))
        .when(num.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _recode_isced_group(pe040: pl.Expr) -> pl.Expr:
    """
    Map PE040 numeric ISCED code → low / medium / high string.
    Uses ISCED_GROUP_BOUNDARIES from constants.
    """
    num = pe040.cast(pl.Float64, strict=False)
    result = pl.lit(None, dtype=pl.String)
    for lo, hi, label in ISCED_GROUP_BOUNDARIES:
        result = (
            pl.when(num.is_between(float(lo), float(hi)))
            .then(pl.lit(label, dtype=pl.String))
            .otherwise(result)
        )
    return result


def _recode_labour_group(pl031: pl.Expr) -> pl.Expr:
    """
    Map PL031 (or PL032) numeric code → employed / unemployed / inactive.
    Uses PL031_TO_LABOUR_GROUP from constants.
    """
    keys   = [float(k) for k in PL031_TO_LABOUR_GROUP.keys()]
    values = list(PL031_TO_LABOUR_GROUP.values())
    return (
        pl031.cast(pl.Float64, strict=False)
        .replace(keys, values, default=None)
        .cast(pl.String)
    )


def _binary_flag(labour_group: pl.Expr, value: str) -> pl.Expr:
    """1.0 if labour_group == value, 0.0 if known-other, null if unknown."""
    return (
        pl.when(labour_group.eq(value)).then(pl.lit(1.0, dtype=pl.Float64))
        .when(labour_group.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


# =============================================================================
# PERSON → HOUSEHOLD AGGREGATION
# Produces one row per household_id with head-level and household-level controls.
# =============================================================================

def _build_person_attributes(
    tr: pl.DataFrame,
    tp: pl.DataFrame,
    year: int,
) -> pl.DataFrame:
    """
    Join Tr + Tp, recode age/sex/education/labour for all persons.
    Returns person-level DataFrame with household_id attached.
    """
    # ── Tr: age + sex ─────────────────────────────────────────────────────────
    tr = tr.with_columns(
        pl.col("RB030").cast(pl.String).alias("person_id"),
        # household_id: DB030 in Tr if present, else derive from person_id
        pl.when(pl.lit("DB030" in tr.columns))
        .then(pl.col("DB030").cast(pl.String))
        .otherwise(
            pl.col("RB030")
            .cast(pl.Int64, strict=False)
            .floordiv(100)
            .cast(pl.String)
        )
        .alias("household_id"),
    )

    # Age: RB082 preferred, then RB081, then year − RB080
    if "RB082" in tr.columns:
        age_expr = pl.col("RB082").cast(pl.Float64, strict=False)
    elif "RB081" in tr.columns:
        age_expr = pl.col("RB081").cast(pl.Float64, strict=False)
    elif "RB080" in tr.columns:
        age_expr = (pl.lit(float(year)) - pl.col("RB080").cast(pl.Float64, strict=False))
    else:
        age_expr = pl.lit(None, dtype=pl.Float64)

    tr = tr.with_columns(
        age_expr.clip(0.0, 110.0).alias("age"),
        pl.col("RB090").cast(pl.Float64, strict=False)
        .replace([1.0, 2.0], ["male", "female"], default=None)
        .cast(pl.String)
        .alias("sex"),
    )

    person = tr.select(["person_id", "household_id", "age", "sex"])

    # ── Tp: education + labour ────────────────────────────────────────────────
    tp = tp.with_columns(
        pl.col("PB030").cast(pl.String).alias("person_id"),
    )

    # Labour: PL031 preferred, PL032 fallback
    if "PL031" in tp.columns:
        lab_raw = pl.col("PL031").cast(pl.Float64, strict=False)
    elif "PL032" in tp.columns:
        lab_raw = pl.col("PL032").cast(pl.Float64, strict=False)
    else:
        lab_raw = pl.lit(None, dtype=pl.Float64)

    tp = tp.with_columns(
        _recode_labour_group(lab_raw).alias("labour_group"),
        _recode_isced_group(
            pl.col("PE040") if "PE040" in tp.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("education_group"),
        (pl.col("PE040").cast(pl.Float64, strict=False)
         if "PE040" in tp.columns else pl.lit(None, dtype=pl.Float64))
        .alias("education_isced"),
    )

    tp_cols = ["person_id", "labour_group", "education_group", "education_isced"]
    tp = tp.select([c for c in tp_cols if c in tp.columns]).unique(subset=["person_id"])

    person = person.join(tp, on="person_id", how="left")

    logger.info("Year %s: person attributes built — %d persons", year, len(person))
    return person


def _aggregate_to_household(person: pl.DataFrame, year: int) -> pl.DataFrame:
    """
    Aggregate person-level attributes to household level.
    Returns one row per household_id with:
      - any_unemployed_hh: 1 if any member has labour_group == "unemployed"
    """
    agg = person.group_by("household_id").agg(
        pl.col("labour_group").eq("unemployed").any().cast(pl.Float64)
        .alias("any_unemployed_hh"),
    )
    logger.info("Year %s: household aggregation — %d households", year, len(agg))
    return agg


def _extract_head_attributes(
    person: pl.DataFrame,
    responsible_ids: pl.DataFrame,   # columns: household_id, responsible_p1_id
) -> pl.DataFrame:
    """
    Join responsible person 1 attributes onto households.
    Returns one row per household with head_* columns.
    """
    head_lookup = person.rename({
        "person_id":        "responsible_p1_id",
        "age":              "head_age",
        "sex":              "head_sex",
        "labour_group":     "head_labour_group",
        "education_group":  "head_education_group",
        "education_isced":  "head_education_isced",
    }).select([
        "responsible_p1_id", "head_age", "head_sex",
        "head_labour_group", "head_education_group", "head_education_isced",
    ])

    head = responsible_ids.join(head_lookup, on="responsible_p1_id", how="left")

    # Binary labour flags for the head
    head = head.with_columns(
        _binary_flag(pl.col("head_labour_group"), "employed").alias("head_employed"),
        _binary_flag(pl.col("head_labour_group"), "unemployed").alias("head_unemployed"),
        _binary_flag(pl.col("head_labour_group"), "inactive").alias("head_inactive"),
    )

    return head.drop("responsible_p1_id")


# =============================================================================
# MAIN YEAR BUILDER
# =============================================================================

def build_household_analysis(
    td: pl.DataFrame,
    th: pl.DataFrame,
    tr: pl.DataFrame | None,
    tp: pl.DataFrame | None,
    year: int,
) -> pl.DataFrame:
    """
    Build one household-year DataFrame for the DiD analysis.

    Parameters
    ----------
    td, th : Polars DataFrames from read_td / read_th_analysis
    tr, tp : Polars DataFrames from read_tr_analysis / read_tp_analysis,
             or None if files are missing (head controls will be null)
    year   : survey year

    Returns
    -------
    Polars DataFrame with one row per household containing:
      household_id, drgn2, region_name, weight_hh, year, post,
      matdep, poverty, income_net_annual, hh_size, equiv_income,
      tenure_status, homeowner,
      head_age, head_sex, head_education_isced, head_education_group,
      head_labour_group, head_employed, head_unemployed, head_inactive,
      any_unemployed_hh
    """
    # ── Validate uniqueness ───────────────────────────────────────────────────
    for key, df, label in [("DB030", td, "Td"), ("HB030", th, "Th")]:
        n_unique = df.select(key).n_unique()
        if n_unique != len(df):
            raise ValueError(
                f"Year {year}: duplicate {key} in {label} "
                f"({len(df)} rows, {n_unique} unique)"
            )

    # ── Td: region + weight ───────────────────────────────────────────────────
    td_clean = td.select(
        pl.col("DB030").cast(pl.String).alias("household_id"),
        pl.col("DB040").cast(pl.String).alias("db040"),
        pl.col("DB090").cast(pl.Float64, strict=False).alias("weight_hh"),
    ).with_columns(
        pl.col("db040")
        .replace(list(DRGN2_MAP.keys()), [float(v) for v in DRGN2_MAP.values()], default=None)
        .cast(pl.Float64)
        .alias("drgn2"),
    ).with_columns(
        pl.col("drgn2")
        .cast(pl.Int32, strict=False)
        .replace(list(REGION_NAMES.keys()), list(REGION_NAMES.values()), default=None)
        .cast(pl.String)
        .alias("region_name"),
    )

    unmapped = td_clean.filter(pl.col("drgn2").is_null()).select("db040").unique()
    if len(unmapped) > 0:
        logger.warning("Year %s: unmapped DB040 values → %s", year, unmapped.to_series().to_list())

    # ── Th: outcomes + controls ───────────────────────────────────────────────
    income_col = "HY020N" if "HY020N" in th.columns else "HY020"
    if income_col not in th.columns:
        logger.warning("Year %s: HY020N and HY020 both absent — income set to null", year)

    th_clean = th.select(
        pl.col("HB030").cast(pl.String).alias("household_id"),
        pl.col("HB080").cast(pl.String).alias("responsible_p1_id")
        if "HB080" in th.columns else pl.lit(None, dtype=pl.String).alias("responsible_p1_id"),

        # ── Outcomes ─────────────────────────────────────────────────────────
        _recode_binary_outcome(
            pl.col("vhMATDEP") if "vhMATDEP" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("matdep"),
        _recode_binary_outcome(
            pl.col("vhPobreza") if "vhPobreza" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("poverty"),
        (pl.col(income_col).cast(pl.Float64, strict=False)
         if income_col in th.columns else pl.lit(None, dtype=pl.Float64))
        .alias("income_net_annual"),

        # ── Controls ─────────────────────────────────────────────────────────
        pl.col("HX040").cast(pl.Float64, strict=False).clip(1.0, None).alias("hh_size")
        if "HX040" in th.columns else pl.lit(None, dtype=pl.Float64).alias("hh_size"),
        pl.col("HX090").cast(pl.Float64, strict=False).alias("equiv_income")
        if "HX090" in th.columns else pl.lit(None, dtype=pl.Float64).alias("equiv_income"),
        pl.col("HH021").cast(pl.Float64, strict=False).alias("tenure_status")
        if "HH021" in th.columns else pl.lit(None, dtype=pl.Float64).alias("tenure_status"),
        _recode_homeowner(
            pl.col("HH021") if "HH021" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("homeowner"),
    )

    if "vhMATDEP" not in th.columns:
        logger.warning("Year %s: vhMATDEP absent", year)
    if "vhPobreza" not in th.columns:
        logger.warning("Year %s: vhPobreza absent", year)

    # ── Merge Td + Th ─────────────────────────────────────────────────────────
    n_th = len(th_clean)
    hh = th_clean.join(td_clean, on="household_id", how="left")
    if len(hh) != n_th:
        raise ValueError(f"Year {year}: row count changed in Td-Th join ({n_th} → {len(hh)})")

    # ── Person-level controls (Tr + Tp) ───────────────────────────────────────
    if tr is not None and tp is not None:
        person = _build_person_attributes(tr, tp, year)

        hh_agg = _aggregate_to_household(person, year)

        responsible_ids = hh_clean = hh.select(["household_id", "responsible_p1_id"]).filter(
            pl.col("responsible_p1_id").is_not_null()
        )
        head_attrs = _extract_head_attributes(person, responsible_ids)

        hh = hh.join(hh_agg,   on="household_id", how="left")
        hh = hh.join(head_attrs, on="household_id", how="left")

    else:
        logger.warning("Year %s: Tr/Tp missing — head controls set to null", year)
        for col, dtype in [
            ("any_unemployed_hh",    pl.Float64),
            ("head_age",             pl.Float64),
            ("head_sex",             pl.String),
            ("head_education_isced", pl.Float64),
            ("head_education_group", pl.String),
            ("head_labour_group",    pl.String),
            ("head_employed",        pl.Float64),
            ("head_unemployed",      pl.Float64),
            ("head_inactive",        pl.Float64),
        ]:
            hh = hh.with_columns(pl.lit(None, dtype=dtype).alias(col))

    # ── Year + Post indicator ─────────────────────────────────────────────────
    hh = hh.with_columns(
        pl.lit(year, dtype=pl.Int32).alias("year"),
        pl.when(pl.lit(year) <= 2019).then(pl.lit(0.0, dtype=pl.Float64))
        .when(pl.lit(year) >= 2021).then(pl.lit(1.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))   # 2020 → null (dropped later)
        .alias("post"),
    )

    logger.info("Year %s: built analysis household frame — %d rows", year, len(hh))
    return hh


# =============================================================================
# PANEL BUILDER
# =============================================================================

def build_analysis_panel(
    years: list[int],
    ecv_dir: Path,
    exposure_path: Path,
) -> pl.DataFrame:
    """
    Build the full analysis panel:
      1. Loop over years, build household frame per year.
      2. Concatenate, drop 2020, exclude regions, check cluster count.
      3. Merge exposure index, construct Post × Exposure interactions.

    Parameters
    ----------
    years        : list of survey years to load (typically ANALYSIS_YEARS)
    ecv_dir      : directory containing ECV_Td_YYYY.dta etc.
    exposure_path: path to output/exposure/exposure_index.csv

    Returns
    -------
    Analysis-ready Polars DataFrame.
    """
    frames: list[pl.DataFrame] = []

    for year in years:
        try:
            td = read_td_analysis(ecv_dir, year)
            th = read_th_analysis(ecv_dir, year)
        except FileNotFoundError as e:
            logger.error("Year %s: %s — skipping", year, e)
            continue

        try:
            tr = read_tr_analysis(ecv_dir, year)
        except FileNotFoundError:
            logger.warning("Year %s: Tr not found — head controls will be null", year)
            tr = None

        try:
            tp = read_tp_analysis(ecv_dir, year)
        except FileNotFoundError:
            logger.warning("Year %s: Tp not found — head controls will be null", year)
            tp = None

        hh = build_household_analysis(td, th, tr, tp, year)
        frames.append(hh)

    if not frames:
        raise RuntimeError("No data loaded — check ecv_dir path and file names")

    panel = pl.concat(frames, how="diagonal")   # diagonal handles occasional column mismatches
    logger.info("Panel before exclusions: %d obs", len(panel))

    # ── Drop 2020 ─────────────────────────────────────────────────────────────
    n_2020 = panel.filter(pl.col("year").eq(2020)).height
    panel = panel.filter(pl.col("year").ne(2020))
    logger.info("Dropped %d obs for year 2020", n_2020)

    # ── Drop missing region ───────────────────────────────────────────────────
    n_no_region = panel.filter(pl.col("drgn2").is_null()).height
    panel = panel.filter(pl.col("drgn2").is_not_null())
    panel = panel.with_columns(pl.col("drgn2").cast(pl.Int32))
    logger.info("Dropped %d obs with no region (Ceuta/Melilla/extra-regio)", n_no_region)

    # ── Drop excluded regions (pre-registered) ────────────────────────────────
    n_before = len(panel)
    panel = panel.filter(~pl.col("drgn2").is_in(list(ANALYSIS_EXCLUDE_DRGN2)))
    logger.info(
        "Dropped %d obs for excluded regions (Aragón=24, La Rioja=23)",
        n_before - len(panel),
    )

    n_clusters = panel.select("drgn2").n_unique()
    if n_clusters != ANALYSIS_N_CLUSTERS:
        logger.warning(
            "Expected %d region clusters, found %d — verify exclusions",
            ANALYSIS_N_CLUSTERS, n_clusters,
        )
    else:
        logger.info("Cluster count: %d ✓", n_clusters)

    # ── Merge exposure index ──────────────────────────────────────────────────
    if not exposure_path.exists():
        raise FileNotFoundError(f"Exposure index not found: {exposure_path}")

    import pandas as pd
    exposure_pd = pd.read_csv(exposure_path)
    exposure = pl.from_pandas(exposure_pd)

    present_specs = [s for s in EXPOSURE_SPECS if s in exposure.columns]
    missing_specs = [s for s in EXPOSURE_SPECS if s not in exposure.columns]
    if missing_specs:
        logger.warning("Specs missing from exposure index: %s", missing_specs)

    exposure = exposure.select(
        [pl.col("drgn2").cast(pl.Int32)] + [pl.col(s) for s in present_specs]
    )

    n_before = len(panel)
    panel = panel.join(exposure, on="drgn2", how="left")
    assert len(panel) == n_before, "Row count changed after exposure merge — check drgn2 keys"

    # ── Post × Exposure interactions ──────────────────────────────────────────
    panel = panel.with_columns([
        (pl.col("post") * pl.col(spec)).alias(f"post_x_{spec}")
        for spec in present_specs
    ])

    logger.info(
        "Exposure merged. Interaction terms: %s",
        [f"post_x_{s}" for s in present_specs],
    )
    logger.info("Final panel: %d obs, %d columns", len(panel), len(panel.columns))
    return panel
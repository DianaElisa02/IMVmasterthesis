"""
ecv_clean.py
============
Household-level ECV cleaning for the IMV DiD analysis pipeline.
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

def _make_path(base: Path, file_type: str, year: int) -> Path:
    prefix_map = {"td": "Td", "th": "Th", "tr": "Tr", "tp": "Tp"}
    prefix = f"ECV_{prefix_map[file_type]}_{year}"
    return base / f"{prefix}.dta"

def read_td_analysis(base: Path, year: int) -> pl.DataFrame:
    from src.constants import TD_COLUMNS
    return _read_section(_make_path(base, "td", year), TD_COLUMNS, year, "td")


def read_th_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "th", year), ANALYSIS_TH_COLUMNS, year, "th_analysis")


def read_tr_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "tr", year), ANALYSIS_TR_COLUMNS, year, "tr_analysis")


def read_tp_analysis(base: Path, year: int) -> pl.DataFrame:
    return _read_section(_make_path(base, "tp", year), ANALYSIS_TP_COLUMNS, year, "tp_analysis")

def _recode_binary_outcome(col: pl.Expr) -> pl.Expr:
    """Map 1→1.0, 0→0.0, anything else→null. Used for VHMATDEP and VHPOBREZA."""
    num = col.cast(pl.Float64, strict=False)
    return (
        pl.when(num.eq(1.0)).then(pl.lit(1.0, dtype=pl.Float64))
        .when(num.eq(0.0)).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _recode_homeowner(hh021: pl.Expr) -> pl.Expr:
    num = hh021.cast(pl.Float64, strict=False)
    return (
        pl.when(num.is_in([1.0, 2.0])).then(pl.lit(1.0, dtype=pl.Float64))
        .when(num.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _recode_isced_group(pe040: pl.Expr) -> pl.Expr:
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
    keys   = [float(k) for k in PL031_TO_LABOUR_GROUP.keys()]
    values = list(PL031_TO_LABOUR_GROUP.values())
    return (
        pl031.cast(pl.Float64, strict=False)
        .replace(keys, values, default=None)
        .cast(pl.String)
    )


def _binary_flag(labour_group: pl.Expr, value: str) -> pl.Expr:
    return (
        pl.when(labour_group.eq(value)).then(pl.lit(1.0, dtype=pl.Float64))
        .when(labour_group.is_not_null()).then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )


def _build_person_attributes(
    tr: pl.DataFrame,
    tp: pl.DataFrame,
    year: int,
) -> pl.DataFrame:
    tr = tr.with_columns(
    pl.col("RB030").cast(pl.String).alias("person_id"),
    pl.col("RB030")
    .cast(pl.Int64, strict=False)
    .floordiv(100)
    .cast(pl.String)
    .alias("household_id"),
    )

    if "RB082" in tr.columns and tr["RB082"].is_not_null().any():
        age_expr = pl.col("RB082").cast(pl.Float64, strict=False)
    elif "RB081" in tr.columns and tr["RB081"].is_not_null().any():
        age_expr = pl.col("RB081").cast(pl.Float64, strict=False)
    elif "RB080" in tr.columns and tr["RB080"].is_not_null().any():
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

    tp = tp.with_columns(
    pl.col("PB030").cast(pl.String).alias("person_id"),
    )

    lab_raw = pl.lit(None, dtype=pl.Float64)
    if "PL031" in tp.columns and tp["PL031"].is_not_null().any():
        lab_raw = pl.col("PL031").cast(pl.Float64, strict=False)
    elif "PL032" in tp.columns:
        lab_raw = pl.col("PL032").cast(pl.Float64, strict=False)

    edu_col = None
    if "PE040" in tp.columns and tp["PE040"].is_not_null().any():
        edu_col = "PE040"
    elif "PE041" in tp.columns and tp["PE041"].is_not_null().any():
        edu_col = "PE041"

    tp = tp.with_columns(
        _recode_labour_group(lab_raw).alias("labour_group"),
        _recode_isced_group(
            pl.col(edu_col) if edu_col else pl.lit(None, dtype=pl.Float64)
        ).alias("education_group"),
        (pl.col(edu_col).cast(pl.Float64, strict=False)
        if edu_col else pl.lit(None, dtype=pl.Float64))
        .alias("education_isced"),
    )
   

    tp_cols = ["person_id", "labour_group", "education_group", "education_isced"]
    tp = tp.select([c for c in tp_cols if c in tp.columns]).unique(subset=["person_id"])
    if "PB220A" in tp.columns:
        tp = tp.with_columns(
            pl.when(
                pl.col("PB220A").cast(pl.Float64, strict=False)
                  .is_in(list(_EU27_PB220A_CODES))
            ).then(pl.lit(0.0))
            .when(pl.col("PB220A").is_not_null())
            .then(pl.lit(1.0))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias("non_eu_born")
        )
    else:
        tp = tp.with_columns(pl.lit(None, dtype=pl.Float64).alias("non_eu_born"))

    tp_cols = ["person_id", "labour_group", "education_group",
               "education_isced", "non_eu_born"]
    tp = tp.select([c for c in tp_cols if c in tp.columns]).unique(subset=["person_id"])

    person = person.join(tp, on="person_id", how="left")
    if "non_eu_born" not in person.columns:
        person = person.with_columns(pl.lit(None, dtype=pl.Float64).alias("non_eu_born"))

    logger.info("Year %s: person attributes built — %d persons", year, len(person))
    return person

_EU27_PB220A_CODES: frozenset[float] = frozenset({
    1.0,   # Spain (own country)
    2.0,   # Germany
    3.0,   # Austria
    4.0,   # Belgium
    5.0,   # Bulgaria
    6.0,   # Cyprus
    7.0,   # Czech Republic
    8.0,   # Denmark
    9.0,   # Estonia
    10.0,  # Finland
    11.0,  # France
    12.0,  # Greece
    13.0,  # Hungary
    14.0,  # Ireland
    15.0,  # Italy
    16.0,  # Latvia
    17.0,  # Lithuania
    18.0,  # Luxembourg
    19.0,  # Malta
    20.0,  # Netherlands
    21.0,  # Poland
    22.0,  # Portugal
    23.0,  # Romania
    24.0,  # Slovakia
    25.0,  # Slovenia
    26.0,  # Sweden
})


def _aggregate_to_household(person: pl.DataFrame, year: int) -> pl.DataFrame:

    ref = (
        person.filter(pl.col("is_reference").eq(True))
        .select([
            "household_id",
            "age",
            "sex",
            "labour_group",
            "education_group",
            "education_isced",
            "non_eu_born",
        ])
        .rename({
            "age":             "head_age",
            "sex":             "head_sex",
            "labour_group":    "head_labour_group",
            "education_group": "head_education_group",
            "education_isced": "head_education_isced",
            "non_eu_born":     "head_non_eu_born",
        })
    )

    ref = ref.with_columns(
        pl.when(pl.col("head_labour_group").eq("unemployed"))
          .then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("head_unemployed"),
        pl.when(pl.col("head_labour_group").eq("employed"))
          .then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("head_employed"),
        pl.when(pl.col("head_education_group").eq("high"))
          .then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("head_high_education"),

        pl.when(pl.col("head_age").lt(35)).then(pl.lit("under35"))
          .when(pl.col("head_age").lt(55)).then(pl.lit("35_54"))
          .when(pl.col("head_age").lt(65)).then(pl.lit("55_64"))
          .otherwise(pl.lit("65plus"))
          .alias("head_age_group"),
    )

    agg = person.group_by("household_id").agg(
        pl.col("labour_group").eq("unemployed").any().cast(pl.Float64)
            .alias("any_unemployed_hh"),
        pl.col("labour_group").eq("employed").any().cast(pl.Float64)
            .alias("any_employed_hh"),
        pl.col("age").lt(18).sum().cast(pl.Float64)
            .alias("n_children"),
        pl.col("age").ge(18).sum().cast(pl.Float64)
            .alias("n_adults"),
        pl.col("sex").eq("female").any().cast(pl.Float64)
            .alias("any_female_hh"),
        pl.col("education_group").eq("high").any().cast(pl.Float64)
            .alias("any_high_education_hh"),
    ).with_columns(
        pl.when(
            pl.col("n_adults").eq(1.0) & pl.col("n_children").ge(1.0)
        ).then(pl.lit(1.0)).otherwise(pl.lit(0.0))
        .alias("single_parent_hh")
    )

    result = agg.join(ref, on="household_id", how="left")
    logger.info("Year: household aggregation — %d households, %d with head attrs",
                len(result), result["head_age"].is_not_null().sum())
    return result
    
def build_household_analysis(
    td: pl.DataFrame,
    th: pl.DataFrame,
    tr: pl.DataFrame | None,
    tp: pl.DataFrame | None,
    year: int,
) -> pl.DataFrame:
    for key, df, label in [("DB030", td, "Td"), ("HB030", th, "Th")]:
        n_unique = df.select(key).n_unique()
        if n_unique != len(df):
            raise ValueError(
                f"Year {year}: duplicate {key} in {label} "
                f"({len(df)} rows, {n_unique} unique)"
            )
 
    td_clean = td.select(
        pl.col("DB030").cast(pl.Int64, strict=False).cast(pl.String).alias("household_id"),
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
 
    if "HY020N" in th.columns and th["HY020N"].is_not_null().any():
        income_col = "HY020N"
    elif "HY020" in th.columns:
        income_col = "HY020"
    else:
        income_col = None
    logger.warning("Year %s: HY020N and HY020 both absent — income set to null", year)
    logger.warning("Year %s: HY020N and HY020 both absent — income set to null", year)
 
    th_clean = th.select(
        pl.col("HB030").cast(pl.Int64, strict=False).cast(pl.String).alias("household_id"),

        _recode_binary_outcome(
            pl.col("VHMATDEP") if "VHMATDEP" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("matdep"),
        _recode_binary_outcome(
            pl.col("VHPOBREZA") if "VHPOBREZA" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("poverty"),
        (pl.col(income_col).cast(pl.Float64, strict=False)
         if income_col in th.columns else pl.lit(None, dtype=pl.Float64))
        .alias("income_net_annual"),

        pl.col("HX040").cast(pl.Float64, strict=False).clip(1.0, None).alias("hh_size")
        if "HX040" in th.columns else pl.lit(None, dtype=pl.Float64).alias("hh_size"),
        pl.col("HX240").cast(pl.Float64, strict=False).alias("equiv_income")
        if "HX240" in th.columns else pl.lit(None, dtype=pl.Float64).alias("equiv_income"),
        pl.col("HH021").cast(pl.Float64, strict=False).alias("tenure_status")
        if "HH021" in th.columns else pl.lit(None, dtype=pl.Float64).alias("tenure_status"),
        _recode_homeowner(
            pl.col("HH021") if "HH021" in th.columns else pl.lit(None, dtype=pl.Float64)
        ).alias("homeowner"),
    )

    if "HB080" in th.columns:
        ref_map = th.select(
            pl.col("HB030").cast(pl.Int64, strict=False).cast(pl.String).alias("household_id"),
            pl.col("HB080").cast(pl.String).alias("ref_person_id"),
        )
    else:
        logger.warning("Year %s: HB080 absent — reference person cannot be identified", year)
        ref_map = None
 
    if "VHMATDEP" not in th.columns:
        logger.warning("Year %s: vhMATDEP absent", year)
    if "VHPOBREZA" not in th.columns:
        logger.warning("Year %s: VHPOBREZA absent", year)
 
    n_th = len(th_clean)
    hh = th_clean.join(td_clean, on="household_id", how="left")
    if len(hh) != n_th:
        raise ValueError(f"Year {year}: row count changed in Td-Th join ({n_th} → {len(hh)})")

    if tr is not None and tp is not None:
        person = _build_person_attributes(tr, tp, year, ref_map=ref_map)
        hh_agg = _aggregate_to_household(person, year)
        hh = hh.join(hh_agg, on="household_id", how="left")
    else:
        logger.warning("Year %s: Tr/Tp missing — household controls set to null", year)
        for col, dtype in [
            ("any_unemployed_hh",    pl.Float64),
            ("any_employed_hh",      pl.Float64),
            ("n_children",           pl.Float64),
            ("n_adults",             pl.Float64),
            ("any_female_hh",        pl.Float64),
            ("any_high_education_hh",pl.Float64),
            ("single_parent_hh",     pl.Float64),
            # Reference-person controls — null when Tr/Tp absent
            ("head_age",             pl.Float64),
            ("head_age_group",       pl.String),
            ("head_unemployed",      pl.Float64),
            ("head_employed",        pl.Float64),
            ("head_high_education",  pl.Float64),
            ("head_non_eu_born",     pl.Float64),
        ]:
            hh = hh.with_columns(pl.lit(None, dtype=dtype).alias(col))

    hh = hh.with_columns(
        pl.lit(year, dtype=pl.Int32).alias("year"),
        pl.when(pl.lit(year) <= 2019).then(pl.lit(0.0, dtype=pl.Float64))
        .when(pl.lit(year) >= 2021).then(pl.lit(1.0, dtype=pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))   
        .alias("post"),
    )
 
    logger.info("Year %s: built analysis household frame — %d rows", year, len(hh))
    return hh


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
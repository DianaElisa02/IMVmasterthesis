"""
constants.py
============
Central repository for all static mappings, recode tables, and column
definitions used in the ECV → EUROMOD UDB conversion pipeline.

All mappings are derived directly from the EUROMOD Input Data Codebook
(EM_data_codebook_J2_0_.xlsm), Spain (ES) sheet, version J2.0+.
Nothing in this module performs computation — it only declares values.
All other modules import from here; nothing here imports from the project.
"""

from __future__ import annotations


YEARS: list[int] = [2017, 2018, 2019]

EUROMOD_DATASET_NAMES: dict[int, str] = {
    2017: "ES_2017_a2",
    2018: "ES_2018_a1",
    2019: "ES_2019_b1",
}


ECV_FILE_PREFIXES: dict[str, str] = {
    "td": "ECV_Td",
    "th": "ECV_Th",
    "tr": "ECV_Tr",
    "tp": "ECV_Tp",
}


TD_COLUMNS: list[str] = [
    "DB030",   # household ID → idhh
    "DB040",   # region code (INE string, e.g. "ES51") → drgn1, drgn2
    "DB090",   # household cross-sectional weight → dwt
    "DB060",   # primary sampling unit → dsu01
    "DB070",   # order of selection of PSU → dsu00
    "DB100",   # degree of urbanisation → drgmd, drgru, drgur
]

TH_COLUMNS: list[str] = [
    "HB030",   # household ID (Th file link key)
    "HH010",   # dwelling type → hh010
    "HH021",   # tenure status → hh021, amrtn
    "HH030",   # number of rooms → hh030, amrrm
    "HH040",   # leaking roof / housing problems → hh040
    "HH050",   # ability to keep home warm → amraw
    "HH060",   # monthly rent → xhcrt component
    "HH070",   # total housing cost → xhc
    "HH071",   # mortgage monthly capital payment → xhcmomc
    "HX040",   # household size → hsize
    "HX090",   # equivalised disposable income → ydses_o
    "HY020",   # total net disposable household income → hy020, yds
    "HY022",   # net income before social transfers excl. pensions → hy022
    "HY023",   # net income before social transfers incl. pensions → hy023
    "HY040G",  # rental income gross → ypr
    "HY050G",  # family/children related allowances gross → bfa
    "HY060G",  # social exclusion benefits gross → bsa
    "HY070G",  # housing allowances gross → bho
    "HY080G",  # inter-household transfers received gross → ypt
    "HY090G",  # investment income gross → yiy
    "HY100G",  # mortgage interest paid gross → xhcmomi
    "HY110G",  # income received by under-16s gross → yot
    "HY120G",  # regular taxes on wealth gross → tpr, twl
    "HY130G",  # regular inter-household transfers paid gross → xmp
    "HY145N",  # repayments/receipts for tax adjustment net → tad
    "HS021",   # arrears on utility bills → amrub
    "HS090",   # computer ownership → aco
    "HS110",   # car ownership → aca
    "HB080",   # person ID of responsible person 1
    "HB090",   # person ID of responsible person 2
    # Note: HY052G, HY053G, HY054G, HY081G, HY140G not present in Spanish ECV.
    # bma, bch00, tintrch, yptmp, tis will be zero in output.
]

TR_COLUMNS: list[str] = [
    "RB030",   # person ID → idperson
    "RB050",   # personal cross-sectional weight
    "RB070",   # month of birth → dmb
    "RB080",   # year of birth (used to compute dag)
    "RB090",   # sex (1=male, 2=female) → dgn
    "RB220",   # father's person ID → idfather
    "RB230",   # mother's person ID → idmother
    "RB240",   # partner's person ID → idpartner
    "RB010",   # survey year (used for dag computation
]

TP_COLUMNS: list[str] = [
    "PB030",   # person ID
    "PB040",   # personal weight
    "PB060",   # weight for selected respondent
    "PB100",   # month of interview → ddt
    "PB110",   # year of interview → ddt
    "PB190",   # legal marital status → dms
    "PE010",   # currently in education (1=yes, 2=no) → dec imputation
    "PE021",   # ISCED level currently attended → dec
    "PE040",   
    "PE041",   # highest ISCED level attended detailed → dehde
    "PL031",   # self-defined economic status (used in ddi fallback)
    "PL032",   # self-defined economic status (alternative) → les, ddi
    "PL040",   # employment status → les fallback
    "PL051",   # occupation ISCO code → loc, lcs
    "PL060",   # hours worked per week in main job → lhw
    "PL073",   # months in full-time employment → liwftmy, liwmy
    "PL074",   # months in part-time employment → liwmy
    "PL075",   # months in full-time self-employment → liwftmy, liwmy
    "PL076",   # months in part-time self-employment → liwmy
    "PL080",   # months in unemployment → lunmy
    "PL085",   # months as pensioner/retired → lpemy, poamy, psumy, pdimy
    "PL086",   # months unable to work due to health → pdimy
    "PL090",   # months in other inactivity → pdimy, poamy
    "PL100",   # hours worked in second job → lhw supplement
    "PL111A",  # NACE industry code (string) → lindi
    "PL200",   # number of years spent in paid work → liwwh
    "PL271",   # duration of most recent unemployment spell → lunwh
    "PY010G",  # employee cash income gross → yem
    "PY020G",  # non-cash employee income gross → kfb
    "PY021G",  # company car gross → kfbcc
    "PY030G",  # employer social insurance contributions gross → tscer
    "PY035G",  # voluntary private pension contributions gross → xpp
    "PY050G",  # self-employment income gross → yse
    "PY080G",  # private pension income gross → ypp
    "PY090G",  # unemployment benefits gross → bun (aggregate only)
    "PY100G",  # old-age benefits gross → poa (aggregate only)
    "PY110G",  # survivor benefits gross → psu (aggregate only)
    "PY120G",  # sickness benefits gross → bhl (aggregate only)
    "PY130G",  # disability benefits gross → pdi (aggregate only)
    "PY140G",  # education allowances gross → bed
    # National SILC breakdown variables (PY092G, PY093G, PY101-103G,
    # PY111-112G, PY122-123G, PY131-133G) are absent from Spanish ECV UDB.
    # EUROMOD simulates sub-components from the aggregates above.
    "PY010N",  # employee cash income net (supplementary)
    "PY050N",  # self-employment income net (supplementary)
    "PB220A",  # nationality → dcz
]


# =============================================================================
# REGION MAPS
# Source: EUROMOD codebook ES sheet, drgn1 and drgn2 derivation notes.
# =============================================================================

# INE NUTS-1 string code → EUROMOD drgn1 integer (1–7 Spain-specific scheme).
# Codebook derivation:
#   drgn1 = 1 if db040 in {ES11, ES12, ES13}       Noroeste
#   drgn1 = 2 if db040 in {ES21, ES22, ES23, ES24}  Noreste
#   drgn1 = 3 if db040 == ES30                       Madrid
#   drgn1 = 4 if db040 in {ES41, ES42, ES43}         Centro
#   drgn1 = 5 if db040 in {ES51, ES52, ES53}         Este
#   drgn1 = 6 if db040 in {ES61, ES62, ES63, ES64}   Sur
#   drgn1 = 7 if db040 == ES70                       Canarias
DRGN1_MAP: dict[str, int] = {
    "ES11": 1, "ES12": 1, "ES13": 1,
    "ES21": 2, "ES22": 2, "ES23": 2, "ES24": 2,
    "ES30": 3,
    "ES41": 4, "ES42": 4, "ES43": 4,
    "ES51": 5, "ES52": 5, "ES53": 5,
    "ES61": 6, "ES62": 6, "ES63": 6, "ES64": 6,
    "ES70": 7,
}

# INE NUTS-1 string code → EUROMOD drgn2 integer.
# Codebook derivation: drgn2 = destring(substr(db040, 3, 2))
# i.e. numeric part of "ES51" → 51.
DRGN2_MAP: dict[str, int] = {
    "ES11": 11, "ES12": 12, "ES13": 13,
    "ES21": 21, "ES22": 22, "ES23": 23, "ES24": 24,
    "ES30": 30,
    "ES41": 41, "ES42": 42, "ES43": 43,
    "ES51": 51, "ES52": 52, "ES53": 53,
    "ES61": 61, "ES62": 62, "ES63": 63, "ES64": 64,
    "ES70": 70,
}

# Mapping from drgn2 to Autonomous Community name.
# Used only for logging and validation output — not fed to EUROMOD.
REGION_NAMES: dict[int, str] = {
    11: "Galicia",
    12: "Principado de Asturias",
    13: "Cantabria",
    21: "País Vasco",
    22: "Comunidad Foral de Navarra",
    23: "La Rioja",
    24: "Aragón",
    30: "Comunidad de Madrid",
    41: "Castilla y León",
    42: "Castilla-La Mancha",
    43: "Extremadura",
    51: "Cataluña",
    52: "Comunitat Valenciana",
    53: "Illes Balears",
    61: "Andalucía",
    62: "Región de Murcia",
    63: "Ciudad de Ceuta",
    64: "Ciudad de Melilla",
    70: "Canarias",
}


# =============================================================================
# EMPLOYMENT STATUS: PL032 → EUROMOD les
# Source: EUROMOD codebook ES sheet, les derivation notes.
#
# EUROMOD Spain derives les primarily from pl032:
#   les = 4  if pl032 == 3   (retired/pensioner)
#   les = 5  if pl032 == 2   (unemployed)
#   les = 6  if pl032 == 5   (student)
#   les = 7  if pl032 == 7   (fulfilling domestic tasks)
#   les = 7  if pl032 == 8   (other inactive)
#   les = 8  if pl032 == 4   (permanently disabled)
#   les = 9  if pl032 == 6   (other)
# Fallback via pl040 when pl032 is missing:
#   les = 2  if pl040 in {1, 2, 4}   (self-employed/employer/family worker)
#   les = 3  if pl040 == 3            (employee)
# =============================================================================

PL031_TO_LES: dict[int, int] = {
    1: 3,   # full-time employee
    2: 3,   # part-time employee
    3: 2,   # full-time self-employed
    4: 2,   # part-time self-employed
    5: 5,   # unemployed
    6: 6,   # student
    7: 4,   # retired
    8: 8,   # permanently disabled
    9: 7,   # domestic tasks → inactive
    10: 7,  # other inactive
}

PL040_TO_LES: dict[int, int] = {
    1: 2,   # employer → self-employed
    2: 2,   # self-employed without employees
    3: 3,   # employee
    4: 2,   # contributing family worker → self-employed
}

LES_DEFAULT: int = 9   # other — used when both pl032 and pl040 are missing


# =============================================================================
# MARITAL STATUS: PB190 → EUROMOD dms
# Source: EUROMOD codebook ES sheet, dms derivation notes.
#
# gen dms = pb190
# recode dms (5=4) (4=5)
# EU-SILC PB190: 1=single, 2=married, 3=separated, 4=divorced, 5=widowed
# EUROMOD dms after recode: 1=single, 2=married, 3=separated, 4=widowed, 5=divorced
# =============================================================================

DMS_RECODE: dict[int, int] = {
    1: 1,   # single → single
    2: 2,   # married → married
    3: 3,   # separated → separated
    4: 5,   # divorced (PB190=4) → divorced (dms=5) after swap
    5: 4,   # widowed (PB190=5) → widowed (dms=4) after swap
}
DMS_DEFAULT: int = 1   # single — used when pb190 is missing and no partner


# =============================================================================
# EDUCATION: PE040/PE041 → EUROMOD deh
# Source: EUROMOD codebook ES sheet, deh derivation notes.
#
# gen deh = pe041
# recode deh (100=1)(200=2)(300/399=3)(400/499=4)(500/800=5)
# =============================================================================

# Boundaries for deh recode — applied as range checks in recode.py.
DEH_RECODE_BOUNDARIES: list[tuple[int, int, int]] = [
    (100, 100, 1),   # primary (PE040=100)
    (200, 200, 2),   # lower secondary (PE040=200)
    (300, 399, 3),   # upper secondary — covers 300, 344, 353, 354
    (400, 499, 4),   # post-secondary non-tertiary — covers 400, 450
    (500, 800, 5),   # tertiary — covers 500
]
DEH_DEFAULT: int = 0   # not completed primary


# =============================================================================
# DISABILITY: PL032 → EUROMOD ddi
# Source: EUROMOD codebook ES sheet, ddi derivation notes.
#
# ddi = 1   if pl032 == 4
# ddi = 0   if pl032 != 4 & pl031 != .
# ddi = -1  if pl032 == . & pb030 == .  (children — not applicable)
# =============================================================================

DDI_DISABLED: int = 1
DDI_NOT_DISABLED: int = 0
DDI_NOT_APPLICABLE: int = -1   # children / information not collected


# =============================================================================
# SEX: RB090 → EUROMOD dgn
# Source: EUROMOD codebook ES sheet, dgn derivation notes.
#
# dgn = rb090   (1=male, 2=female — no recode for Spain)
# =============================================================================

DGN_VALID_VALUES: frozenset[int] = frozenset({1, 2})
DGN_DEFAULT: int = 1


# =============================================================================
# NACE INDUSTRY: PL111A → EUROMOD lindi
# Source: EUROMOD codebook ES sheet, lindi derivation notes.
# PL111A in Spanish ECV is a lowercase string letter code.
# =============================================================================

LINDI_MAP: dict[str, int] = {
    "a":     1,    # agriculture and fishing
    "b - e": 2,    # mining, manufacturing and utilities
    "f":     3,    # construction
    "g":     4,    # wholesale and retail
    "i":     5,    # hotels and restaurants
    "h":     6,    # transport
    "j":     6,    # communication (grouped with transport)
    "k":     7,    # financial intermediation
    "l - n": 8,    # real estate and business
    "o":     9,    # public administration and defence
    "p":     10,   # education
    "q":     11,   # health and social work
    "r - u": 12,   # other services
}
LINDI_DEFAULT: int = 0   # not applicable


# =============================================================================
# UDB OUTPUT COLUMN ORDER
# Defines the column order in the output .txt file.
# Columns not produced by this pipeline default to 0 inside EUROMOD.
# Order follows the codebook variable groupings.
# =============================================================================

UDB_COLUMN_ORDER: list[str] = [
    # identifiers
    "idhh", "idperson", "idmother", "idfather", "idpartner",
    # demographic
    "dag", "dgn", "dct", "dcz", "ddi", "deh", "dms", "dmb",
    "dwt", "drgn1", "drgn2", "drgmd", "drgru", "drgur",
    "dsu00", "dsu01",
    # labour market
    "les", "lhw", "lse", "lcs", "loc", "lindi",
    "liwmy", "liwftmy", "liwptmy", "liwwh",
    "lunmy", "lunwh", "lpemy",
    # household structure
    "hsize", "oecd_m",
    # housing
    "hh010", "hh021", "hh030", "hh040",
    # household income
    "hy020", "hy022", "hy023",
    "yds", "yiy", "ypr", "ypt", "yptmp", "yot",
    # personal income
    "yem", "yse", "ypp", "kfb", "kfbcc",
    # unemployment benefits
    "bun", "bunct", "bunnc", "bunot",
    # health benefits
    "bhl", "bhl00", "bhlot",
    # disability benefits
    "pdi", "pdi00", "pdicm", "pdinc", "pdiot",
    # old-age benefits
    "poa", "poa00", "poacm", "poanc",
    # survivor benefits
    "psu", "psuwd00", "psuwdcm",
    # other benefits
    "bed", "bfa", "bch", "bch00", "bchdi", "bchot",
    "bho", "bma", "bsa",
    # taxes and expenditures
    "tad", "tis", "tpr", "tscer", "twl",
    "xhc", "xhcmomi", "xhcmomc", "xhcrt", "xhcot",
    "xmp", "xpp",
    # assets
    "amrtn", "amrrm", "amraw", "amrub", "aca", "aco", "afc",
]

OUTPUT_SEPARATOR: str = "\t"
OUTPUT_MISSING_VALUE: str = "0"   # EUROMOD uses 0 as default for missing inputs
OUTPUT_ENCODING: str = "utf-8"

EUROMOD_OUTPUT_DIR: Path = Path("input_data/euromod_output")
EXPOSURE_OUTPUT_DIR: Path = Path("output/exposure")

# Pre-reform RMI simulation outputs (Run A)
RMI_FILES: dict[int, Path] = {
    2017: EUROMOD_OUTPUT_DIR / "es_2017_std.txt",
    2018: EUROMOD_OUTPUT_DIR / "es_2018_std.txt",
    2019: EUROMOD_OUTPUT_DIR / "es_2019_std.txt",
}

# IMV counterfactual simulation outputs (Run B — 2022 rules)
IMV_FILES: dict[int, Path] = {
    2017: EUROMOD_OUTPUT_DIR / "IMV_2022ruleson2017.txt",
    2018: EUROMOD_OUTPUT_DIR / "IMV_2022ruleson2018.txt",
    2019: EUROMOD_OUTPUT_DIR / "IMV_2022ruleson2019.txt",
}

# Regions excluded from exposure computation
# La Rioja (23) and Aragón (24): broken RMI parameterisation in both
EXPOSURE_EXCLUDE_REGIONS: frozenset[int] = frozenset({23, 24})

RMI_INCOMPATIBLE_REGIONS: frozenset[int] = frozenset({
    11,  # Galicia
    53,  # Illes Balears
    61,  # Andalucía
})

RMI_COMPLEMENTARY_REGIONS: frozenset[int] = frozenset({
    12, 13, 21, 22, 30, 41, 42, 43,
    51, 52, 62, 63, 64, 70,
})

# IMV statutory amounts (2022) — used for validation bounds
# Source: Law 19/2021, updated 2022
IMV_STATUTORY_2022: dict[str, float] = {
    "basic_monthly":          469.93,
    "increment_per_member":   0.30,
    "max_multiplier":         2.20,
    "max_monthly":            1033.85,
    "lone_parent_supplement": 0.22,
    "floor_monthly":          10.0,
}

# IMV administrative benchmarks (2022 national)
# Source: INSS Estadística IMV 2022
IMV_ADMIN_2022: dict[str, float] = {
    "recipients":      603_000,
    "expenditure_M":   2_100.0,
    "mean_monthly":    290.0,
}
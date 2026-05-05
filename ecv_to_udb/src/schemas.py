"""
schemas.py
==========
Pandera schemas validating the final merged UDB output before export.

Two schemas are defined:
- PersonUdbSchema: validates the person-level UDB DataFrame before export.
- HouseholdUdbSchema: validates household-level variables merged onto persons.

Constraints are derived from the EUROMOD Input Data Codebook (ES sheet, J2.0+)
and apply only to variables this pipeline actively populates. Variables set to
zero by default (absent from Spanish ECV UDB) are not constrained here.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema


PersonUdbSchema = DataFrameSchema(
    {
        # --- identifiers ---
        "idhh": Column(nullable=False),
        "idperson": Column(nullable=False, unique=True),

        # --- demographic ---
        "dag": Column(
            float,
            checks=[Check.ge(0), Check.le(120)],
            nullable=True,
        ),
        "dgn": Column(
            float,
            checks=Check.isin([1.0, 2.0]),
            nullable=True,
        ),
        "dct": Column(
            float,
            checks=Check.equal_to(13.0),
            nullable=False,
        ),
        "ddi": Column(
            float,
            checks=Check.isin([-1.0, 0.0, 1.0]),
            nullable=True,
        ),
        "deh": Column(
            float,
            checks=[Check.ge(0.0), Check.le(5.0)],
            nullable=True,
        ),
        "dms": Column(
            float,
            checks=[Check.ge(1.0), Check.le(5.0)],
            nullable=True,
        ),
        "dwt": Column(
            float,
            checks=Check.ge(0.0),
            nullable=True,
        ),
        "drgn1": Column(
            float,
            checks=[Check.ge(1.0), Check.le(7.0)],
            nullable=True,
        ),
        "drgn2": Column(
            float,
            checks=Check.isin([
                11.0, 12.0, 13.0,
                21.0, 22.0, 23.0, 24.0,
                30.0,
                41.0, 42.0, 43.0,
                51.0, 52.0, 53.0,
                61.0, 62.0, 63.0, 64.0,
                70.0,
            ]),
            nullable=True,
        ),

        # --- labour market ---
        "les": Column(
            float,
            checks=[Check.ge(0.0), Check.le(9.0)],
            nullable=True,
        ),
        "lhw": Column(
            float,
            checks=[Check.ge(0.0), Check.le(80.0)],
            nullable=True,
        ),
        "liwmy": Column(
            float,
            checks=[Check.ge(0.0), Check.le(12.0)],
            nullable=True,
        ),
        "lunmy": Column(
            float,
            checks=[Check.ge(0.0), Check.le(12.0)],
            nullable=True,
        ),
        "lpemy": Column(
            float,
            checks=[Check.ge(0.0), Check.le(12.0)],
            nullable=True,
        ),

        # --- household structure ---
        "hsize": Column(
            float,
            checks=[Check.ge(1.0), Check.le(20.0)],
            nullable=True,
        ),
        "oecd_m": Column(
            float,
            checks=Check.ge(1.0),
            nullable=True,
        ),

        # --- household income ---
        "hy020": Column(float, nullable=True),
        "hy022": Column(float, nullable=True),
        "hy023": Column(float, nullable=True),

        # --- personal income (monthly gross, can be negative for self-employment) ---
        "yem": Column(float, checks=Check.ge(0.0), nullable=True),
        "yse": Column(float, nullable=True),
        "ypp": Column(float, checks=Check.ge(0.0), nullable=True),

        # --- benefits (non-negative monthly amounts) ---
        "bun": Column(float, checks=Check.ge(0.0), nullable=True),
        "bhl": Column(float, checks=Check.ge(0.0), nullable=True),
        "pdi": Column(float, checks=Check.ge(0.0), nullable=True),
        "poa": Column(float, checks=Check.ge(0.0), nullable=True),
        "psu": Column(float, checks=Check.ge(0.0), nullable=True),
        "bed": Column(float, checks=Check.ge(0.0), nullable=True),
        "bsa": Column(float, checks=Check.ge(0.0), nullable=True),
        "bfa": Column(float, checks=Check.ge(0.0), nullable=True),
        "bho": Column(float, checks=Check.ge(0.0), nullable=True),
    },
    strict=False,
    coerce=True,
)


HouseholdUdbSchema = DataFrameSchema(
    {
        "idhh": Column(nullable=False, unique=True),
        "drgn1": Column(
            float,
            checks=[Check.ge(1.0), Check.le(7.0)],
            nullable=True,
        ),
        "drgn2": Column(float, nullable=True),
        "dwt": Column(float, checks=Check.ge(0.0), nullable=True),
        "hsize": Column(float, checks=Check.ge(1.0), nullable=True),
        "hy020": Column(float, nullable=True),
        "hh021": Column(float, nullable=True),
        "hh030": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)
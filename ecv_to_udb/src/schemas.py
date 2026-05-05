"""
schemas.py
==========
Pandera/Polars schemas validating the final merged UDB output before export.
"""

from __future__ import annotations

import polars as pl
import pandera.polars as pa

PersonUdbSchema = pa.DataFrameSchema(
    {
        "idhh":     pa.Column(pl.String, nullable=False),
        "idperson": pa.Column(pl.String, nullable=False, unique=True),

        "dag":   pa.Column(pl.Float64, checks=[pa.Check.ge(0), pa.Check.le(120)], nullable=True),
        "dgn":   pa.Column(pl.Float64, checks=pa.Check.isin([1.0, 2.0]), nullable=True),
        "dct":   pa.Column(pl.Float64, checks=pa.Check.equal_to(13.0), nullable=False),
        "ddi":   pa.Column(pl.Float64, checks=pa.Check.isin([-1.0, 0.0, 1.0]), nullable=True),
        "deh":   pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(5.0)], nullable=True),
        "dms":   pa.Column(pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(5.0)], nullable=True),
        "dwt":   pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "drgn1": pa.Column(pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(7.0)], nullable=True),
        "drgn2": pa.Column(
            pl.Float64,
            checks=pa.Check.isin([
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

        "les":   pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(9.0)], nullable=True),
        "lhw":   pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(80.0)], nullable=True),
        "liwmy": pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True),
        "lunmy": pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True),
        "lpemy": pa.Column(pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True),

        "hsize":  pa.Column(pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(20.0)], nullable=True),
        "oecd_m": pa.Column(pl.Float64, checks=pa.Check.ge(1.0), nullable=True),

        "hy020": pa.Column(pl.Float64, nullable=True),
        "hy022": pa.Column(pl.Float64, nullable=True),
        "hy023": pa.Column(pl.Float64, nullable=True),

        "yem": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "yse": pa.Column(pl.Float64, nullable=True),
        "ypp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),

        "bun": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bhl": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "pdi": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "poa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "psu": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bed": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bsa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bfa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bho": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
    },
    strict=False,
    coerce=True,
)


HouseholdUdbSchema = pa.DataFrameSchema(
    {
        "idhh":  pa.Column(pl.String, nullable=False, unique=True),
        "drgn1": pa.Column(pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(7.0)], nullable=True),
        "drgn2": pa.Column(pl.Float64, nullable=True),
        "dwt":   pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hsize": pa.Column(pl.Float64, checks=pa.Check.ge(1.0), nullable=True),
        "hy020": pa.Column(pl.Float64, nullable=True),
        "hh021": pa.Column(pl.Float64, nullable=True),
        "hh030": pa.Column(pl.Float64, nullable=True),
    },
    strict=False,
    coerce=True,
)

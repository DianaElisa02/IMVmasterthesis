from __future__ import annotations

import pandera.polars as pa
import polars as pl

PersonUdbSchema = pa.DataFrameSchema(
    {
        # ── Identifiers ───────────────────────────────────────────────────────
        "idhh": pa.Column(pl.String, nullable=False),
        "idperson": pa.Column(pl.String, nullable=False, unique=True),
        # ── Demographics ──────────────────────────────────────────────────────
        "dag": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0), pa.Check.le(120)], nullable=True
        ),
        "dgn": pa.Column(pl.Float64, checks=pa.Check.isin([1.0, 2.0]), nullable=True),
        "dct": pa.Column(pl.Float64, checks=pa.Check.equal_to(13.0), nullable=False),
        "dcz": pa.Column(
            pl.Float64, checks=pa.Check.isin([1.0, 2.0, 3.0]), nullable=True
        ),
        "ddi": pa.Column(
            pl.Float64, checks=pa.Check.isin([-1.0, 0.0, 1.0]), nullable=True
        ),
        "deh": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(5.0)], nullable=True
        ),
        "dms": pa.Column(
            pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(5.0)], nullable=True
        ),
        "dmb": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        # ── Weights and geography ─────────────────────────────────────────────
        "dwt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "drgn1": pa.Column(
            pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(7.0)], nullable=True
        ),
        "drgn2": pa.Column(
            pl.Float64,
            checks=pa.Check.isin(
                [
                    11.0, 12.0, 13.0,
                    21.0, 22.0, 23.0, 24.0,
                    30.0,
                    41.0, 42.0, 43.0,
                    51.0, 52.0, 53.0,
                    61.0, 62.0, 63.0, 64.0,
                    70.0,
                ]
            ),
            nullable=True,
        ),
        "drgmd": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "drgru": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "drgur": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "dsu00": pa.Column(

            pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True
        ),
        "dsu01": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Labour market ─────────────────────────────────────────────────────
        "les": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(9.0)], nullable=True
        ),
        "lhw": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(80.0)], nullable=True
        ),
        "lse": pa.Column(
            pl.Float64, checks=pa.Check.isin([0.0, 1.0, 2.0, 3.0]), nullable=True
        ),
        "lcs": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "loc": pa.Column(
            pl.Float64, checks=[pa.Check.ge(-1.0), pa.Check.le(9.0)], nullable=True
        ),
        "lindi": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        "liwmy": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        "liwftmy": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        "liwptmy": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        "liwwh": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(780.0)], nullable=True
        ),
        "lunmy": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        "lpemy": pa.Column(
            pl.Float64, checks=[pa.Check.ge(0.0), pa.Check.le(12.0)], nullable=True
        ),
        # ── Household structure ───────────────────────────────────────────────
        "hsize": pa.Column(
            pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(20.0)], nullable=True
        ),
        "oecd_m": pa.Column(pl.Float64, checks=pa.Check.ge(1.0), nullable=True),

        "hh010": pa.Column(
            pl.Float64,
            checks=pa.Check.isin([1.0, 2.0, 3.0, 4.0]),
            nullable=True,
        ),
        "hh021": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hh030": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hh040": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Household income (net figures may be negative) ────────────────────
        "hy020": pa.Column(pl.Float64, nullable=True),
        "hy022": pa.Column(pl.Float64, nullable=True),
        "hy023": pa.Column(pl.Float64, nullable=True),
        "yds": pa.Column(pl.Float64, nullable=True),
        "yiy": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "ypr": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "ypt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "yot": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Personal income ───────────────────────────────────────────────────
        "yem": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "yse": pa.Column(pl.Float64, nullable=True),  # losses possible
        "ypp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "kfb": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "kfbcc": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Benefits ──────────────────────────────────────────────────────────
        "bun": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bhl": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "pdi": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "poa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "psu": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bed": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bsa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bfa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bho": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Taxes and expenditure ─────────────────────────────────────────────
        "tad": pa.Column(pl.Float64, nullable=True),  # net adjustment, can be negative
        "tpr": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "tscer": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "twl": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhc": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcmomi": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcrt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcot": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xmp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xpp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        # ── Assets ────────────────────────────────────────────────────────────
        "amrrm": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "amraw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "amrub": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "aca": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "aco": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
    },
    strict=False,
    coerce=True,
)


HouseholdUdbSchema = pa.DataFrameSchema(
    {
        "IDHH": pa.Column(pl.String, nullable=False, unique=True),
        "dct": pa.Column(pl.Float64, checks=pa.Check.equal_to(13.0), nullable=False),
        "drgn1": pa.Column(
            pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(7.0)], nullable=True
        ),
        "drgn2": pa.Column(
            pl.Float64,
            checks=pa.Check.isin(
                [
                    11.0, 12.0, 13.0,
                    21.0, 22.0, 23.0, 24.0,
                    30.0,
                    41.0, 42.0, 43.0,
                    51.0, 52.0, 53.0,
                    61.0, 62.0, 63.0, 64.0,
                    70.0,
                ]
            ),
            nullable=True,
        ),
        "drgmd": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "drgru": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "drgur": pa.Column(pl.Float64, checks=pa.Check.isin([0.0, 1.0]), nullable=True),
        "dwt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hsize": pa.Column(
            pl.Float64, checks=[pa.Check.ge(1.0), pa.Check.le(20.0)], nullable=True
        ),
        "hh010": pa.Column(
            pl.Float64,
            checks=pa.Check.isin([1.0, 2.0, 3.0, 4.0]),
            nullable=True,
        ),
        "hh021": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hh030": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hh040": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "hy020": pa.Column(pl.Float64, nullable=True),
        "hy022": pa.Column(pl.Float64, nullable=True),
        "hy023": pa.Column(pl.Float64, nullable=True),
        "yds": pa.Column(pl.Float64, nullable=True),
        "yiy": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "ypr": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "ypt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "yot": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bfa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bsa": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "bho": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "tad": pa.Column(pl.Float64, nullable=True),
        "tpr": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "twl": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhc": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcmomi": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcrt": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xhcot": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "xmp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "amrrm": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "amraw": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "amrub": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "aca": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
        "aco": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=True),
    },
    strict=False,
    coerce=True,
)
import gzip
from pathlib import Path
from typing import NamedTuple, Any
import datetime as dt
from itertools import groupby
from common.io import (
    ParamIndex,
    read_fcs_metadata,
    FCSHeader,
    TEXT_HEADER,
    AnyTEXT,
    ParamKeyword,
    ParsedParam,
)
from common.functional import fmap_maybe

# X = TypeVar("X")
# Y = TypeVar("Y")
# Z = TypeVar("Z")


# MONTHMAP = {
#     k: i + 1
#     for i, k, in enumerate(
#         [
#             "jan",
#             "feb",
#             "mar",
#             "apr",
#             "may",
#             "jun",
#             "jul",
#             "aug",
#             "sep",
#             "oct",
#             "nov",
#             "dec",
#         ]
#     )
# }


# # TODO these might make more sense as pydantic classes
# class Param(NamedTuple):
#     param_index: int  # the 'n' in PnX
#     shortname: str  # PnN
#     bits: int  # PnB
#     maxrange: float  # PnR
#     log_decades: float  # PnE (field 1)
#     log_zero: float  # PnE (field 2)
#     longname: str | None  # PnS
#     filtername: str | None  # PnF
#     gain: float | None  # PnG
#     wavelength: float | None  # PnL
#     power: float | None  # PnO
#     percent_emitted: float | None  # PnP
#     detector_type: str | None  # PnT
#     detector_voltage: float | None  # PnV


class FCSCMeta(NamedTuple):
    file_index: int
    org: str
    machine: str
    material: str
    sop: int
    exp: int
    rep: int
    group: str
    om: str
    filepath: Path


class FCSMeta(NamedTuple):
    fcsc: FCSCMeta
    header: FCSHeader
    standard: AnyTEXT
    params: dict[ParamIndex, ParsedParam]
    nonstandard: dict[str, str]
    total_time: float | None
    warnings: list[str]


# def parse_date(s: str) -> str:
#     """
#     Parse the date in each cytometer to standardized format.

#     Each cytometer (unfortunately) saves their date in a slightly different
#     format. Some are like YYYY-MM-DD, others like DD-MM-YYYY, different
#     cases for month, etc.

#     Sad face.
#     """
#     ss = s.split("-")
#     year_is_first = len(ss[0]) == 4
#     year_n = int(ss[0 if year_is_first else 2])
#     day_n = int(ss[2 if year_is_first else 0])
#     month_n = MONTHMAP[ss[1].lower()]
#     return dt.date(year_n, month_n, day_n).isoformat()


def parse_params(ps: list[ParamKeyword]) -> dict[ParamIndex, ParsedParam]:
    grouped = groupby(sorted(ps, key=lambda x: x.index_), key=lambda x: x.index_)
    return {
        g[0]: ParsedParam.parse_obj({k: str(v) for _, k, v in g[1]}) for g in grouped
    }


def parse_total_time(meta: AnyTEXT) -> tuple[str | None, str | None, float | None]:
    btime = fmap_maybe(dt.time.fromisoformat, meta.btim)
    etime = fmap_maybe(dt.time.fromisoformat, meta.etim)
    if btime is not None and etime is not None:
        offset = dt.timedelta(days=0 if btime < etime else 1)
        begin = dt.datetime.combine(dt.date.today(), btime)
        end = dt.datetime.combine(dt.date.today() + offset, etime)
        total = (end - begin).total_seconds()
    else:
        total = None
    return (
        fmap_maybe(lambda x: x.isoformat(), btime),
        fmap_maybe(lambda x: x.isoformat(), etime),
        total,
    )


def parse_group(sop: int, exp: int, material: str) -> str:
    if sop == 1:
        return "SOP 1"
    elif sop == 2:
        n = "3/4" if exp == 4 else str(exp)
        return f"SOP 2: Matrix {n}"
    else:
        if "fmo" in material:
            s = "Test FMO"
        elif exp == 1:
            s = "Test Pheno"
        elif exp == 2:
            s = "Test Count"
        elif exp == 3:
            s = "QC Count"
        else:
            s = "QC Pheno"
        return f"SOP 3: {s}"


def parse_metadata(idx: int, p: Path) -> FCSMeta:
    s = p.name.split("_")
    org = s[2]
    machine = s[3]
    material = s[4]
    sop = int(s[5][5])
    exp = int(s[6][1])
    fcsc = FCSCMeta(
        file_index=idx,
        org=org,
        machine=machine,
        material=material,
        sop=sop,
        exp=exp,
        group=parse_group(sop, exp, material),
        om=f"{org}_{machine}",
        rep=int(s[8]),
        filepath=p,
    )

    res = read_fcs_metadata(p)
    params = parse_params(res.meta.params)
    btime, etime, total_time = parse_total_time(res.meta.standard)
    return FCSMeta(
        fcsc,
        res.meta.header,
        res.meta.standard,
        params,
        res.meta.nonstandard,
        total_time,
        res.warnings,
    )


def to_tsv_line(xs: list[str]) -> str:
    return "\t".join(xs) + "\n"


def main(smk: Any) -> None:
    root = Path(smk.input[0])

    meta_out = Path(smk.output["meta"])
    params_out = Path(smk.output["params"])
    nonstandard_out = Path(smk.output["nonstandard"])
    warnings_out = Path(smk.output["warnings"])

    allmeta = [
        parse_metadata(i, p)
        for i, p in enumerate(sorted(root.iterdir()))
        if p.name.endswith(".fcs")
    ]

    with gzip.open(meta_out, "wt") as f:
        header = [
            *FCSCMeta._fields,
            *FCSHeader._fields,
            *TEXT_HEADER,
            "total_time",
        ]
        f.write(to_tsv_line(header))
        for m in allmeta:
            xs = (
                [*m.fcsc, *m.header._asdict().values()]
                + m.standard.line
                + [m.total_time]
            )
            f.write(to_tsv_line(xs))

    with gzip.open(params_out, "wt") as f:
        header = [
            *FCSCMeta._fields,
            "param_index",
            *ParsedParam.header(),
        ]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for pi, pd in m.params.items():
                xs = [*m.fcsc, pi, *pd.line]
                f.write(to_tsv_line(xs))

    with gzip.open(nonstandard_out, "wt") as f:
        header = [*FCSCMeta._fields, "key", "value"]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for k, v in m.nonstandard.items():
                xs = [*m.fcsc, k, v]
                f.write(to_tsv_line(xs))

    with gzip.open(warnings_out, "wt") as f:
        header = [*FCSCMeta._fields, "warning"]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for w in m.warnings:
                xs = [*m.fcsc, pi, w]
                f.write(to_tsv_line(xs))


main(snakemake)  # type: ignore

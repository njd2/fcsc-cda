import re
import warnings
import gzip
import fcsparser as fp  # type: ignore
from pathlib import Path
from typing import NamedTuple, TypeVar, Callable, Any
import datetime as dt
from itertools import groupby

X = TypeVar("X")
Y = TypeVar("Y")
Z = TypeVar("Z")


MONTHMAP = {
    k: i + 1
    for i, k, in enumerate(
        [
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
        ]
    )
}


# TODO these might make more sense as pydantic classes
class Param(NamedTuple):
    param_index: int  # the 'n' in PnX
    shortname: str  # PnN
    bits: int  # PnB
    maxrange: float  # PnR
    log_decades: float  # PnE (field 1)
    log_zero: float  # PnE (field 2)
    longname: str | None  # PnS
    filtername: str | None  # PnF
    gain: float | None  # PnG
    wavelength: float | None  # PnL
    power: float | None  # PnO
    percent_emitted: float | None  # PnP
    detector_type: str | None  # PnT
    detector_voltage: float | None  # PnV


class FCSMeta(NamedTuple):
    file_index: int
    org: str
    machine: str
    material: str
    sop: int
    exp: int
    rep: int
    group: str
    om: str
    # header fields
    version: float
    header_text0: int
    header_text1: int
    header_data0: int
    header_data1: int
    header_analysis0: int
    header_analysis1: int
    # required keywords
    total: int
    analysis0: int
    analysis1: int
    stext0: int
    stext1: int
    data0: int
    data1: int
    datatype: str
    byteord: str
    mode: str
    nextdata: int
    filepath: Path
    params: list[Param]
    # optional keywords
    timestep: float | None
    btime: str | None
    etime: str | None
    volume: float | None
    run_date: str | None
    total_time: float | None
    operator: str | None
    serial: str | None
    lost: int | None
    aberrant: int | None
    comment: str | None
    cytometer: str | None
    investigator: str | None
    institution: str | None
    system: str | None
    description: str | None
    warnings: list[str]


def lookup_maybe(k: X, xs: dict[X, Y]) -> Y | None:
    return xs[k] if k in xs else None


def fmap_maybe(f: Callable[[X], Y], x: X | None) -> Y | None:
    return f(x) if x is not None else None


def lookup_map(f: Callable[[Y], Z | None], k: X, xs: dict[X, Y]) -> Z | None:
    return fmap_maybe(f, lookup_maybe(k, xs))


def float_maybe(x: str) -> float | None:
    try:
        return float(x)
    except ValueError:
        return None


def parse_date(s: str) -> str:
    """
    Parse the date in each cytometer to standardized format.

    Each cytometer (unfortunately) saves their date in a slightly different
    format. Some are like YYYY-MM-DD, others like DD-MM-YYYY, different
    cases for month, etc.

    Sad face.
    """
    ss = s.split("-")
    year_is_first = len(ss[0]) == 4
    year_n = int(ss[0 if year_is_first else 2])
    day_n = int(ss[2 if year_is_first else 0])
    month_n = MONTHMAP[ss[1].lower()]
    return dt.date(year_n, month_n, day_n).isoformat()


def parse_params(meta: dict[str, str]) -> list[Param]:
    params = [
        (int(m[1]), m[2], v)
        for k, v in meta.items()
        if (m := re.match("\\$P([0-9]+)([BEFGLNOPRSTV])$", k))
    ]
    grouped = groupby(sorted(params, key=lambda x: x[0]), key=lambda x: x[0])
    return [
        Param(
            param_index=g[0],
            shortname=(xs := {k: v for _, k, v in g[1]})["N"].strip(),
            bits=int(xs["B"]),
            maxrange=int(xs["R"]),
            log_decades=float((e := xs["E"].split(","))[0]),
            log_zero=float(e[1]),
            longname=lookup_maybe("S", xs),
            filtername=lookup_maybe("F", xs),
            gain=lookup_map(float, "G", xs),
            wavelength=lookup_map(float_maybe, "L", xs),
            power=lookup_map(float, "O", xs),
            percent_emitted=lookup_map(float, "P", xs),
            detector_type=lookup_maybe("T", xs),
            detector_voltage=lookup_map(float_maybe, "V", xs),
        )
        for g in grouped
    ]


def parse_total_time(
    meta: dict[str, str]
) -> tuple[str | None, str | None, float | None]:
    btime = lookup_map(dt.time.fromisoformat, "$BTIM", meta)
    etime = lookup_map(dt.time.fromisoformat, "$ETIM", meta)
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
    # try:
    with warnings.catch_warnings(record=True, action="always") as w:
        meta = fp.parse(p, meta_data_only=True)
        warn_msgs = [str(x.message).replace("\n", " ") for x in w]
    # except UserWarning as e:
    # raise Exception(p.name, e)
    header = meta["__header__"]
    params = parse_params(meta)
    btime, etime, total_time = parse_total_time(meta)
    org = s[2]
    machine = s[3]
    material = s[4]
    sop = int(s[5][5])
    exp = int(s[6][1])
    group = parse_group(sop, exp, material)
    om = f"{org}_{machine}"
    return FCSMeta(
        file_index=idx,
        filepath=p,
        org=org,
        machine=machine,
        om=om,
        group=group,
        material=material,
        sop=sop,
        exp=exp,
        rep=int(s[8]),
        # header
        version=float(header["FCS format"].decode()[3:]),
        header_text0=header["text start"],
        header_text1=header["text end"],
        header_data0=header["data start"],
        header_data1=header["data end"],
        header_analysis0=header["analysis start"],
        header_analysis1=header["analysis end"],
        # text segment
        stext0=int(meta["$BEGINSTEXT"]),
        stext1=int(meta["$ENDSTEXT"]),
        data0=int(meta["$BEGINDATA"]),
        data1=int(meta["$ENDDATA"]),
        analysis0=int(meta["$BEGINANALYSIS"]),
        analysis1=int(meta["$ENDANALYSIS"]),
        datatype=meta["$DATATYPE"],
        nextdata=meta["$NEXTDATA"],
        mode=meta["$MODE"],
        byteord=meta["$BYTEORD"],
        total=int(meta["$TOT"]),
        timestep=lookup_map(float, "$TIMESTEP", meta),
        btime=btime,
        etime=etime,
        total_time=total_time,
        volume=lookup_map(float, "$VOL", meta),
        run_date=lookup_map(parse_date, "$DATE", meta),
        operator=lookup_maybe("$OP", meta),
        serial=lookup_maybe("$CYTSN", meta),
        lost=lookup_map(int, "$LOST", meta),
        aberrant=lookup_map(int, "$ABRT", meta),
        comment=lookup_maybe("$COMMENT", meta),
        cytometer=lookup_maybe("$CYT", meta),
        investigator=lookup_maybe("$EXP", meta),
        institution=lookup_maybe("$INST", meta),
        system=lookup_maybe("$SYS", meta),
        description=lookup_maybe("$CELLS", meta),
        params=params,
        warnings=warn_msgs,
    )


def to_tsv_line(xs: list[str]) -> str:
    return "\t".join(xs) + "\n"


def main(smk: Any) -> None:
    root = Path(smk.input[0])

    meta_out = Path(smk.output["meta"])
    params_out = Path(smk.output["params"])
    warnings_out = Path(smk.output["warnings"])

    allmeta = [
        parse_metadata(i, p)
        for i, p in enumerate(sorted(root.iterdir()))
        if p.name.endswith(".fcs")
    ]

    with gzip.open(meta_out, "wt") as f:
        header = [k for k in FCSMeta._fields if k not in ["params", "warnings"]]
        f.write(to_tsv_line(header))
        for m in allmeta:
            xs = [
                str(v) if v is not None else ""
                for k, v in m._asdict().items()
                if k not in ["params", "warnings"]
            ]
            f.write(to_tsv_line(xs))

    with gzip.open(params_out, "wt") as f:
        header = ["file_index", "org", "machine", "om", "group", *Param._fields]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for p in m.params:
                xs = [
                    str(m.file_index),
                    m.org,
                    m.machine,
                    m.om,
                    m.group,
                    *["" if y is None else str(y) for y in p._asdict().values()],
                ]
                f.write(to_tsv_line(xs))

    with gzip.open(warnings_out, "wt") as f:
        header = ["file_index", "org", "machine", "om", "group", "warning"]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for w in m.warnings:
                xs = [str(m.file_index), m.org, m.machine, m.om, m.group, w]
                f.write(to_tsv_line(xs))


main(snakemake)  # type: ignore

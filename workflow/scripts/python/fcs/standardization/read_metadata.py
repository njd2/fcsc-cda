from pathlib import Path
from typing import NamedTuple, Any
import datetime as dt
from common.metadata import split_path, FCSMeta, FileIndex
from common.io import (
    group_params,
    _TEXT31,
    StandardKey,
    Version,
    ParamIndex,
    read_fcs_metadata,
    FCSHeader,
    TABULAR_TEXT_HEADER,
    TabularTEXT,
    ParsedParam,
)


class AllMeta(NamedTuple):
    fcs: FCSMeta
    fcs_header: FCSHeader
    standard: TabularTEXT
    params: dict[ParamIndex, ParsedParam]
    nonstandard: dict[str, str]
    deviant: dict[StandardKey, str]
    total_time: float | None
    warnings: list[str]

    @property
    def tsv_header(self) -> list[str]:
        return [*FCSMeta._fields, *FCSHeader._fields, *AllMeta._fields[2:]]


# def parse_params(ps: list[ParamKeyword]) -> dict[ParamIndex, ParsedParam]:
#     grouped = groupby(sorted(ps, key=lambda x: x.index_), key=lambda x: x.index_)
#     return {
#         g[0]: ParsedParam.parse_obj({k: str(v) for _, k, v in g[1]}) for g in grouped
#     }


def parse_total_time(meta: TabularTEXT) -> tuple[str | None, str | None, float | None]:
    if meta.btim is not None and meta.etim is not None:
        offset = dt.timedelta(days=0 if meta.btim < meta.etim else 1)
        begin = dt.datetime.combine(dt.date.today(), meta.btim)
        end = dt.datetime.combine(dt.date.today() + offset, meta.etim)
        total = (end - begin).total_seconds()
    else:
        total = None
    return (str(meta.btim), str(meta.etim), total)


def parse_metadata(idx: int, p: Path) -> AllMeta:
    fcs = FCSMeta(file_index=FileIndex(idx), filemeta=split_path(p), filepath=p)
    res = read_fcs_metadata(p)
    params = group_params(res.meta.params)
    btime, etime, total_time = parse_total_time(res.meta.standard)
    return AllMeta(
        fcs,
        res.header,
        res.meta.standard,
        params,
        res.meta.nonstandard,
        res.meta.deviant,
        total_time,
        res.warnings,
    )


def to_tsv_line(xs: list[str]) -> str:
    return "\t".join(xs) + "\n"


def main(smk: Any) -> None:
    file_paths = Path(smk.input["fcs_files"])

    meta_out = Path(smk.output["meta"])
    params_out = Path(smk.output["params"])
    nonstandard_out = Path(smk.output["nonstandard"])
    warnings_out = Path(smk.output["warnings"])

    with open(file_paths, "r") as f:
        allmeta = [
            parse_metadata(int((ss := x.rstrip().split("\t"))[0]), Path(ss[1]))
            for x in f
        ]

    with open(meta_out, "wt") as f:
        header = [
            *FCSMeta.tsv_header(),
            *FCSHeader._fields,
            *TABULAR_TEXT_HEADER,
            "total_time",
            "version",
        ]
        f.write(to_tsv_line(header))
        for m in allmeta:
            version = (
                Version.v3_1 if isinstance(m.standard, _TEXT31) else Version.v3_0
            ).value
            sm = m.standard.mapping
            xs = [
                *m.fcs.line,
                *m.fcs_header.line,
                *[sm[x] if x in sm else "" for x in TABULAR_TEXT_HEADER],
                "" if m.total_time is None else str(m.total_time),
                version,
            ]
            f.write(to_tsv_line(xs))

    with open(params_out, "wt") as f:
        header = [
            *FCSMeta.tsv_header(),
            "param_index",
            *ParsedParam.header(),
        ]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for pi, pd in m.params.items():
                xs = [*m.fcs.line, str(pi), *pd.line]
                f.write(to_tsv_line(xs))

    with open(nonstandard_out, "wt") as f:
        header = [*FCSMeta.tsv_header(), "deviant", "key", "value"]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for k, v in m.nonstandard.items():
                xs = [
                    *m.fcs.line,
                    "False",
                    k,
                    v.replace("\n", "\\n").replace("\t", "\\t"),
                ]
                f.write(to_tsv_line(xs))
            for k, v in m.deviant.items():
                xs = [
                    *m.fcs.line,
                    "True",
                    k,
                    v.replace("\n", "\\n").replace("\t", "\\t"),
                ]
                f.write(to_tsv_line(xs))

    with open(warnings_out, "wt") as f:
        header = [*FCSMeta.tsv_header(), "warning"]
        f.write(to_tsv_line(header))
        for m in allmeta:
            for w in m.warnings:
                xs = [*m.fcs.line, w]
                f.write(to_tsv_line(xs))


main(snakemake)  # type: ignore

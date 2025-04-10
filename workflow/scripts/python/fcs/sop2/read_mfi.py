import csv
import pandas as pd
from pathlib import Path
from typing import Any, NamedTuple
from flowkit import parse_gating_xml, Sample  # type: ignore
from common.io import read_fcs
import common.metadata as ma
import common.sop2 as s2
from multiprocessing import Pool


class MFIResult(NamedTuple):
    primary: ma.Color
    peak_index: int
    bead_count: int
    pop_count: int
    raw_mfi: dict[ma.Color, float]


class Meta(NamedTuple):
    fcs: ma.FcsCompensationMeta
    gate_path: Path


class Line(NamedTuple):
    meta: Meta
    mfis: list[MFIResult]
    matrix: ma.Matrix | None


def read_mfi(m: Meta) -> Line:
    gstrat = parse_gating_xml(m.gate_path)
    fcs_path = m.fcs.indexed_path.filepath
    parsed = read_fcs(fcs_path)
    df_raw = parsed.events
    s = Sample(df_raw, sample_id=str(fcs_path.name))
    res = gstrat.gate_sample(s)
    bead_count = res.get_gate_membership("beads").sum()

    def go(gate_name: str) -> MFIResult:
        s = gate_name.split("_")
        primary = ma.Color(s[0])
        i = int(s[1])
        mask = res.get_gate_membership(gate_name)
        df_msk = df_raw[mask]
        return MFIResult(
            primary,
            i,
            bead_count,
            mask.sum(),
            {c: df_msk[c.value].median() for c in ma.Color},
        )

    mfi = [go(g[0]) for g in gstrat.get_child_gate_ids("beads")]
    return Line(m, mfi, m.fcs.filemeta.matrix)


def main(smk: Any) -> None:
    files_path = Path(smk.input["files"])
    m1_out = Path(smk.output["m1"])
    m2_out = Path(smk.output["m2"])
    m3_out = Path(smk.output["m3"])

    df_files = pd.read_table(files_path)

    runs = [
        Meta(s, Path(gates))
        for file_index, fcs, gates in df_files.itertuples(index=False)
        if isinstance(
            (
                s := ma.split_indexed_path(
                    ma.IndexedPath(ma.FileIndex(int(file_index)), Path(fcs))
                )
            ).filemeta,
            ma.CompensationMeta,
        )
    ]

    with Pool(smk.threads) as pl:
        results = pl.map(read_mfi, runs[0:10])

    m1, m2, m3 = s2.partition_by_matrix(lambda x: x.matrix, results)

    def to_df(xs: list[Line]) -> pd.DataFrame:
        header = [*ma.IndexedPath._fields, "om", "primary", "secondary"]
        df = pd.DataFrame(
            [
                (
                    *x.meta.fcs.indexed_path,
                    x.meta.fcs.filemeta.machine.om,
                    *m,
                )
                for x in xs
                for m in x.mfis
                for secondary, mfi in m.raw_mfi.items()
            ],
            columns=header,
        )
        df_cal = df.pivot(
            index=["om", "std_name"],
            columns=["peak_index"],
            values=["raw_mfi"],
        )
        df_cal.columns = pd.Index(["auto", "stained"])
        return df

    # df_fc = to_df(fs)
    # df_rainbow = to_df(rs)

    # df_cal.columns = pd.Index(["p0", "p1"])
    # df_cal = df_cal.join(df_erf, on=["om", "std_name"]).reset_index()
    # df_cal["slope"] = df_cal["erf"] / (df_cal["p1"] - df_cal["p0"])
    # # anything with only one peak will be NaN for slope
    # df_cal = df_cal.dropna(subset=["slope"])

    # df_fc.to_csv(fc_out, sep="\t", index=False)
    # df_rainbow.to_csv(rainbow_out, sep="\t", index=False)
    # df_cal.to_csv(cal_out, sep="\t", index=False)


main(snakemake)  # type: ignore

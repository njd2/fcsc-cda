import pandas as pd
from pathlib import Path
from typing import Any, NamedTuple
from flowkit import parse_gating_xml, Sample  # type: ignore
from common.io import read_fcs
from common.functional import partition
from common.metadata import Color, OM, IndexedPath, FileIndex, split_path
import common.sop1 as s1
from multiprocessing import Pool


class MFIResult(NamedTuple):
    std_name: str
    peak_index: int
    bead_count: int
    pop_count: int
    raw_mfi: float


class Meta(NamedTuple):
    fcs: IndexedPath
    gate_path: Path
    om: OM


class Line(NamedTuple):
    meta: Meta
    mfis: list[MFIResult]
    israinbow: bool


def read_mfi(m: Meta) -> Line:
    color = s1.path_to_color(m.fcs.filepath)
    gstrat = parse_gating_xml(m.gate_path)
    fcs = read_fcs(m.fcs.filepath)
    df_raw = fcs.events
    s = Sample(df_raw, sample_id=str(m.fcs.filepath.name))
    res = gstrat.gate_sample(s)
    bead_count = res.get_gate_membership("beads").sum()

    def go(gate_name: str) -> MFIResult:
        s = gate_name.split("_")
        std_name = Color(s[0]).value
        i = int(s[1])
        mask = res.get_gate_membership(gate_name)
        raw_mfi = df_raw[mask][std_name].median()
        return MFIResult(std_name, i, bead_count, mask.sum(), raw_mfi)

    if color is None:
        # rainbow beads: there should only be one color channel
        gs = [g[0] for g in gstrat.get_child_gate_ids("beads")]
        if len(gs) == 0:
            # this should never happen, but just in case ;)
            mfi = []
        else:
            masks = {int(g.split("_")[1]): res.get_gate_membership(g) for g in gs}
            # indices are numbered starting from the bottom, but the bottom
            # peaks are also the most likely to be missed, so start counting
            # from top down starting at 7 (to 0)
            max_index = max(masks.keys())
            mfi = [
                MFIResult(
                    c.value,
                    i + 7 - max_index,
                    bead_count,
                    mask.sum(),
                    df_raw[mask][c.value].median(),
                )
                for c in Color
                for i, mask in masks.items()
            ]
        return Line(m, mfi, True)
    else:
        # fc beads
        mfi = [go(g[0]) for g in gstrat.get_child_gate_ids("beads")]
        return Line(m, mfi, False)


def main(smk: Any) -> None:
    files_path = Path(smk.input["files"])
    erf_path = Path(smk.input["erf"])
    fc_out = Path(smk.output["fc"])
    rainbow_out = Path(smk.output["rainbow"])
    cal_out = Path(smk.output["cal"])

    df_files = pd.read_table(files_path)
    df_erf = (pd.read_table(erf_path).set_index(["om", "std_name"])).drop(
        columns=["channel"]
    )

    runs = [
        Meta(
            IndexedPath(FileIndex(int(file_index)), (p := Path(fcs))),
            Path(gates),
            split_path(p).machine.om,
        )
        for file_index, fcs, gates in df_files.itertuples(index=False)
    ]

    with Pool(smk.threads) as pl:
        results = pl.map(read_mfi, runs)

    rs, fs = partition(lambda x: x.israinbow, results)

    def to_df(xs: list[Line]) -> pd.DataFrame:
        header = [*IndexedPath._fields, "om", *MFIResult._fields]
        return pd.DataFrame(
            [(*x.meta.fcs, x.meta.om, *m) for x in xs for m in x.mfis],
            columns=header,
        )

    df_fc = to_df(fs)
    df_rainbow = to_df(rs)

    df_cal = df_fc.pivot(
        index=["om", "std_name"],
        columns=["peak_index"],
        values=["raw_mfi"],
    )

    df_cal.columns = pd.Index(["p0", "p1"])
    df_cal = df_cal.join(df_erf, on=["om", "std_name"]).reset_index()
    df_cal["slope"] = df_cal["erf"] / (df_cal["p1"] - df_cal["p0"])
    # anything with only one peak will be NaN for slope
    df_cal = df_cal.dropna(subset=["slope"])

    df_fc.to_csv(fc_out, sep="\t", index=False)
    df_rainbow.to_csv(rainbow_out, sep="\t", index=False)
    df_cal.to_csv(cal_out, sep="\t", index=False)


main(snakemake)  # type: ignore

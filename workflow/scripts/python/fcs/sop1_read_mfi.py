import csv
import pandas as pd
from pathlib import Path
from typing import Any, NamedTuple
from flowkit import parse_gating_xml, Sample  # type: ignore
from common.io import read_fcs
import common.sop1 as s1
from multiprocessing import Pool


class MFIResult(NamedTuple):
    gcolor: s1.Color
    peak_index: int
    bead_count: int
    raw_mfi: float
    xform_mfi: float


class Meta(NamedTuple):
    file_index: int
    fcs_path: Path
    gate_path: Path
    om: s1.OM


class Line(NamedTuple):
    meta: Meta
    mfis: list[MFIResult]
    israinbow: bool


def read_mfi(m: Meta) -> Line:
    color = s1.path_to_color(m.fcs_path)
    gstrat = parse_gating_xml(m.gate_path)
    fcs = read_fcs(m.fcs_path)
    df_raw = fcs.events
    s = Sample(df_raw, sample_id=str(m.fcs_path.name))
    res = gstrat.gate_sample(s)
    trans = {c: gstrat.get_transform(f"{c}_logicle") for c in s1.COLORS}
    s.apply_transform(trans)
    df_x = s.as_dataframe(source="xform")

    def go(gate_name: str) -> MFIResult:
        s = gate_name.split("_")
        gcolor = s1.Color(s[0])
        i = int(s[1])
        mask = res.get_gate_membership(gate_name)
        raw_mfi = df_raw[mask][gcolor].median()
        x_mfi = df_x[mask][gcolor].median()
        return MFIResult(gcolor, i, mask.sum(), raw_mfi, x_mfi)

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
                    c,
                    i + 7 - max_index,
                    mask.sum(),
                    df_raw[mask][c].median(),
                    df_x[mask][c].median(),
                )
                for c in s1.COLORS
                for i, mask in masks.items()
            ]
        return Line(m, mfi, True)
    else:
        # fc beads
        mfi = [go(g[0]) for g in gstrat.get_child_gate_ids("beads")]
        return Line(m, mfi, False)


def main(smk: Any) -> None:
    files_path = Path(smk.input[0])
    fc_out = Path(smk.output["fc"])
    rainbow_out = Path(smk.output["rainbow"])

    df = pd.read_table(files_path)

    runs = [
        Meta(int(file_index), (p := Path(fcs)), Path(gates), s1.path_to_om(p))
        for file_index, fcs, gates in df.itertuples(index=False)
    ]

    with Pool(smk.threads) as pl:
        results = pl.map(read_mfi, runs)

    with open(fc_out, "wt") as fo, open(rainbow_out, "wt") as ro:
        fw = csv.writer(fo, delimiter="\t")
        rw = csv.writer(ro, delimiter="\t")
        header = [*Meta._fields, *MFIResult._fields]
        fw.writerow(header)
        rw.writerow(header)
        for x in results:
            for m in x.mfis:
                row = [*x.meta, *m]
                if x.israinbow:
                    rw.writerow(row)
                else:
                    fw.writerow(row)


main(snakemake)  # type: ignore

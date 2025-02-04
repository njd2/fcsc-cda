import csv
import pandas as pd
from itertools import groupby
from pathlib import Path
from typing import Any, NamedTuple
from flowkit import parse_gating_xml, Sample  # type: ignore
from common.io import read_fcs
import common.sop1 as s1
from common.functional import partition
from multiprocessing import Pool


class MFIResult(NamedTuple):
    gcolor: s1.Color
    peak_index: int
    raw_mfi: float
    xform_mfi: float


class Meta(NamedTuple):
    file_index: int
    fcs_path: Path
    gate_path: Path
    om: s1.OM


class FCLine(NamedTuple):
    meta: Meta
    color: s1.Color
    mfis: list[MFIResult]


class RainbowLine(NamedTuple):
    meta: Meta
    mfis: list[MFIResult]


def read_mfi(m: Meta) -> FCLine | RainbowLine:
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
        return MFIResult(gcolor, i, raw_mfi, x_mfi)

    # indices are numbered starting from the bottom, but the bottom peaks are
    # also the most likely to be missed, so start counting from top down
    # starting at 7 (to 0)
    def fix_indices(xs: list[MFIResult]) -> list[MFIResult]:
        maxi = max(x[1] for x in xs)
        return [MFIResult(c, i + 7 - maxi, r, x) for (c, i, r, x) in xs]

    if color is None:
        gs = [g[0] for g in gstrat.get_child_gate_ids("beads")]
        mfi = [
            y
            for _, xs in groupby((go(g) for g in gs), lambda x: x[0])
            for y in fix_indices(list(xs))
        ]
        return RainbowLine(m, mfi)
    else:
        mfi = [go(g[0]) for g in gstrat.get_child_gate_ids("beads")]
        return FCLine(m, color, mfi)


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

    fs, rs = partition(lambda x: isinstance(x, FCLine), results)

    with open(fc_out, "wt") as fo, open(rainbow_out, "wt") as ro:
        metaheaders = Meta._fields
        mfiheaders = MFIResult._fields
        fw = csv.writer(fo, delimiter="\t")
        rw = csv.writer(ro, delimiter="\t")
        fw.writerow([*metaheaders, "color", *mfiheaders])
        rw.writerow([*metaheaders, *mfiheaders])
        for x in results:
            for m in x.mfis:
                if isinstance(x, FCLine):
                    fw.writerow([*x.meta, x.color, *m])
                else:
                    rw.writerow([*x.meta, *m])


main(snakemake)  # type: ignore

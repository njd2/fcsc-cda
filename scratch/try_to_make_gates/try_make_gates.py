from pathlib import Path
import flowkit as fk  # type: ignore
import fcsparser as fp  # type: ignore
from typing import Any, NamedTuple
import pandas as pd
from bokeh.plotting import show
from bokeh.layouts import row, column
from bokeh.plotting import figure, show
from bokeh.models import TabPanel, Tabs

RAINBOW_PAT = "RCP-30"

META_COLUMNS = {
    "file_index": int,
    "om": str,
    "sop": int,
    "filepath": str,
}

NO_SCATTER = [
    "LMNXSEA_CellStream-1",
    "LMNXSEA_CellStream-2",
    "LMNXSEA_ImageStreamX-1",
]


class GateRanges(NamedTuple):
    fsc_min: int
    fsc_max: int
    ssc_min: int
    ssc_max: int


def make_gates(path: Path, gs: GateRanges) -> Any:
    _, df = fp.parse(path, channel_naming="$PnN")
    smp = fk.Sample(df, sample_id=str(path))

    dim_fsc = fk.Dimension("fsc_a", range_min=gs.fsc_min, range_max=gs.fsc_max)
    dim_ssc = fk.Dimension("ssc_a", range_min=gs.ssc_min, range_max=gs.ssc_max)

    rect_top_left_gate = fk.gates.RectangleGate(
        "beads",
        dimensions=[dim_fsc, dim_ssc],
    )

    g_strat = fk.GatingStrategy()
    g_strat.add_gate(rect_top_left_gate, gate_path=("root",))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=mask,
    )
    p1 = smp.plot_scatter(
        "ac7",
        "fsc_a",
        source="raw",
        highlight_mask=mask,
    )
    return row(p0, p1)


def read_paths(path: Path) -> dict[str, list[Path]]:
    df = pd.read_table(path, usecols=[*META_COLUMNS], dtype=META_COLUMNS)
    df = df[df["sop"] == 1].drop(columns=["sop", "file_index"])

    return {
        str(om): [*map(Path, data["filepath"].tolist())]
        for om, data in df.groupby("om")
        if om not in NO_SCATTER
    }


path_map = read_paths(
    Path("results/intermediate/meta/standardized_fcs/standard_metadata.tsv")
)


def write_blank_gate_ranges(ip: Path, op: Path) -> None:
    path_map = read_paths(ip)
    no_rainbow = [(om, False, 0, 1e12, 0, 1e12) for om in path_map]
    rainbow = [(om, True, 0, 1e12, 0, 1e12) for om in path_map]
    df = pd.DataFrame(
        no_rainbow + rainbow,
        columns=[
            "om",
            "is_rainbow",
            "fsc_min",
            "fsc_max",
            "ssc_min",
            "ssc_max",
        ],
    )
    df.to_csv(op, index=False, sep="\t")


def read_gate_ranges(p: Path) -> dict[tuple[str, bool], GateRanges]:
    df = pd.read_table(p)
    return {
        (om, is_rainbow): GateRanges(int(f0), int(f1), int(s0), int(s1))
        for om, is_rainbow, f0, f1, s0, s1 in df.itertuples(index=False)
    }


# write_blank_gate_ranges(
#     Path("results/intermediate/meta/standardized_fcs/standard_metadata.tsv"),
#     Path("static/sop1_gates.tsv"),
# )


def make_plots(om: str) -> None:
    all_gs = read_gate_ranges(Path("static/sop1_gates.tsv"))
    path_map = read_paths(
        Path("results/intermediate/meta/standardized_fcs/standard_metadata.tsv")
    )
    paths = path_map[om]

    non_rainbow_tab = TabPanel(
        child=column(
            *[
                make_gates(p, all_gs[(om, False)])
                for p in paths
                if RAINBOW_PAT not in p.name
            ]
        ),
        title="Non Rainbow",
    )
    rainbow_plot = next(
        (make_gates(p, all_gs[(om, True)]) for p in paths if RAINBOW_PAT in p.name),
        None,
    )
    if rainbow_plot:
        rainbow_tab = TabPanel(child=column(rainbow_plot), title="Rainbow")
        page = Tabs(tabs=[non_rainbow_tab, rainbow_tab])
    else:
        page = Tabs(tabs=[non_rainbow_tab])
    show(page)


make_plots("WRAIR_Fortessa")

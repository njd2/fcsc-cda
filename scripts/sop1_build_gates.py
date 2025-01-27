#! /usr/bin/env python3

import math
from tempfile import NamedTemporaryFile
import argparse
import sys
import os
from pathlib import Path
import flowkit as fk  # type: ignore
import fcsparser as fp  # type: ignore
from typing import Any, NamedTuple, IO, NewType
import pandas as pd
import numpy as np
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs

OM = NewType("OM", str)

RAINBOW_PAT = "RCP-30"

NO_SCATTER = [
    OM("LMNXSEA_CellStream-1"),
    OM("LMNXSEA_CellStream-2"),
    OM("LMNXSEA_ImageStreamX-1"),
]

COLORS = ["v450", "v500", "fitc", "pc55", "pe", "pc7", "apc", "ac7"]

FILE_ORDER = [
    f"FC-{x}_SOP"
    for x in [
        "V450",
        "V500-C",
        "FITC",
        "PerCP-Cy5.5",
        "PE",
        "PE-Cy7",
        "APC",
        "APC-Cy7",
    ]
]

LogicleM = 4.5


class GateRanges(NamedTuple):
    fsc_min: int
    fsc_max: int
    ssc_min: int
    ssc_max: int


def build_gating_strategy(gs: GateRanges) -> Any:
    dim_fsc = fk.Dimension("fsc_a", range_min=gs.fsc_min, range_max=gs.fsc_max)
    dim_ssc = fk.Dimension("ssc_a", range_min=gs.ssc_min, range_max=gs.ssc_max)

    rect_top_left_gate = fk.gates.RectangleGate(
        "beads",
        dimensions=[dim_fsc, dim_ssc],
    )

    g_strat = fk.GatingStrategy()
    g_strat.add_gate(rect_top_left_gate, gate_path=("root",))

    return g_strat


def apply_gates_to_sample(fcs_path: Path, gs: GateRanges, colors: list[str]) -> Any:
    g_strat = build_gating_strategy(gs)
    _, df = fp.parse(fcs_path, channel_naming="$PnN")
    smp = fk.Sample(df, sample_id=str(fcs_path.name))

    fsc_max = df["fsc_a"].max()
    ssc_max = df["ssc_a"].max()

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=mask,
        x_max=min(gs.fsc_max * 5, fsc_max),
        y_max=min(gs.ssc_max * 5, ssc_max),
    )

    # p1 = smp.plot_scatter(
    #     "fsc_h",
    #     "ssc_h",
    #     source="raw",
    #     highlight_mask=mask,
    # )

    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    smp_beads = fk.Sample(df_beads, sample_id=str(fcs_path.name))
    color_max = max([df_beads[c].max() for c in COLORS])
    trans = {}
    # see Parks et al (the Logicle Paper) for formulas/rationale for doing this
    for c in COLORS:
        arr = df_beads[c].values
        arr_neg = arr[arr < 0]
        if arr_neg.size < 10:
            trans[c] = fk.transforms.LogicleTransform(color_max, 1.0, LogicleM, 0)
        else:
            low_ref = np.quantile(arr_neg, 0.05)
            best_W = (LogicleM - math.log(color_max / abs(low_ref))) / 2
            trans[c] = fk.transforms.LogicleTransform(color_max, best_W, LogicleM, 0)
    smp_beads.apply_transform(trans)
    ps = [
        smp_beads.plot_histogram(c, source="xform", x_range=(-0.2, 1)) for c in colors
    ]
    return row(p0, *ps)


def read_path_map(files_path: Path) -> dict[OM, list[Path]]:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    df = pd.read_table(files_path, usecols=[1], names=["file_path"])
    acc: dict[OM, list[Path]] = {}
    for _, x in df["file_path"].items():
        p = Path(x)
        xs = p.name.split("_")
        om = OM(f"{xs[2]}_{xs[3]}")
        if xs[5] == "SOP-01" and om not in NO_SCATTER:
            if om not in acc:
                acc[om] = []
            acc[om] += [p]
    for k in acc:
        acc[k].sort(
            key=lambda x: next((i for i, o in enumerate(FILE_ORDER) if o in x.name), -1)
        )
    return acc


# def write_blank_gate_ranges(files_path: Path) -> None:
#     path_map = read_path_map(files_path)
#     no_rainbow = [(om, False, 0, 1e12, 0, 1e12) for om in path_map]
#     rainbow = [(om, True, 0, 1e12, 0, 1e12) for om in path_map]
#     df = pd.DataFrame(
#         no_rainbow + rainbow,
#         columns=[
#             "om",
#             "is_rainbow",
#             "fsc_min",
#             "fsc_max",
#             "ssc_min",
#             "ssc_max",
#         ],
#     )
#     df.to_csv(sys.stdout, index=False, sep="\t")


def read_gate_ranges(ranges_path: Path) -> dict[tuple[OM, bool], GateRanges]:
    df = pd.read_table(ranges_path)
    return {
        (om, is_rainbow): GateRanges(int(f0), int(f1), int(s0), int(s1))
        for om, is_rainbow, f0, f1, s0, s1 in df.itertuples(index=False)
    }


def make_plots(
    ranges_path: Path,
    files_path: Path,
    om: OM,
    rainbow: bool,
    colors: list[str],
) -> None:
    all_gs = read_gate_ranges(ranges_path)
    path_map = read_path_map(files_path)
    paths = path_map[om]

    non_rainbow_tab = TabPanel(
        child=column(
            *[
                apply_gates_to_sample(p, all_gs[(om, False)], colors)
                for p in paths
                if RAINBOW_PAT not in p.name
            ]
        ),
        title="Non Rainbow",
    )
    rainbow_plot = next(
        (
            apply_gates_to_sample(p, all_gs[(om, True)], colors)
            for p in paths
            if RAINBOW_PAT in p.name
        ),
        None,
    )
    if rainbow_plot:
        rainbow_tab = TabPanel(child=column(rainbow_plot), title="Rainbow")
        page = Tabs(
            tabs=(
                [rainbow_tab, non_rainbow_tab]
                if rainbow
                else [non_rainbow_tab, rainbow_tab]
            )
        )
    else:
        page = Tabs(tabs=[non_rainbow_tab])
    # open but don't close temp file to save plot
    tf = NamedTemporaryFile(suffix="_sop1.html", delete_on_close=False, delete=False)
    output_file(tf.name)
    show(page)


def write_gate(
    ranges_path: Path,
    om: OM,
    gate_handle: IO[bytes],
    rainbow: bool,
) -> None:
    all_gs = read_gate_ranges(ranges_path)
    gs = build_gating_strategy(all_gs[(om, rainbow)])
    fk.export_gatingml(gs, gate_handle)


def write_all_gates(
    files_path: Path,
    ranges_path: Path,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_map = read_path_map(files_path)
    for om in path_map:

        def go(rainbow: bool) -> None:
            r = "rcp" if rainbow else "fc"
            with open(out_dir / f"{prefix}_{str(om)}_{r}.xml", "wb") as ho:
                write_gate(ranges_path, om, ho, rainbow)

        go(True)
        go(False)


def list_oms(files_path: Path) -> None:
    xs = read_path_map(files_path)
    for x in xs:
        print(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    list_parser = subparsers.add_parser("list", help="list all org/machine IDs")
    list_parser.add_argument("files", help="path to list of files")

    plot_parser = subparsers.add_parser("plot", help="plot gates for org/machine ID")
    plot_parser.add_argument("om", help="org/machine ID")
    plot_parser.add_argument("files", help="path to list of files")
    plot_parser.add_argument("gates", help="path to list of gate ranges")
    plot_parser.add_argument(
        "-R",
        "--rainbow",
        action="store_true",
        help="show rainbow first (for extreme laziness)",
    )
    plot_parser.add_argument(
        "-c",
        "--colors",
        help="comma separated list of colors to include",
    )

    write_gate_parser = subparsers.add_parser(
        "write_gate",
        help="write gate for org/machine ID",
    )
    write_gate_parser.add_argument("om", help="org/machine ID")
    write_gate_parser.add_argument("gates", help="path to list of gate ranges")
    write_gate_parser.add_argument(
        "-R",
        "--rainbow",
        action="store_true",
        help="save rainbow gate",
    )

    write_gates_parser = subparsers.add_parser("write_gates", help="write all gates")
    write_gates_parser.add_argument("files", help="path to list of files")
    write_gates_parser.add_argument("gates", help="path to list of gate ranges")
    write_gates_parser.add_argument("outdir", help="output directory")
    write_gates_parser.add_argument("prefix", help="output file prefix ")

    parsed = parser.parse_args(sys.argv[1:])

    if parsed.cmd == "list":
        list_oms(Path(parsed.files))

    if parsed.cmd == "plot":
        make_plots(
            Path(parsed.gates),
            Path(parsed.files),
            OM(parsed.om),
            parsed.rainbow,
            COLORS if parsed.colors is None else parsed.colors.split(","),
        )

    if parsed.cmd == "write_gate":
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            write_gate(Path(parsed.gates), OM(parsed.om), f, parsed.rainbow)

    if parsed.cmd == "write_gates":
        write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.outdir),
            parsed.prefix,
        )


main()

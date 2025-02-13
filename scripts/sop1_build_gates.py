#! /usr/bin/env python3

import json
import argparse
import sys
from typing import Any
from tempfile import NamedTemporaryFile
from pathlib import Path
from flowkit import Sample  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs
from common.functional import fmap_maybe, unzip2, partition
from common.metadata import Color, OM
import common.sop1 as s1
import common.gating as ga
import common.metadata as ma


def apply_gates_to_sample(
    gs: ga.SOP1Gates,
    color_ranges: ma.RangeMap,
    fcs: ma.FcsCalibrationMeta,
    colors: list[Color],
    scatteronly: bool,
) -> tuple[Any, ga.GatingStrategyDebug]:
    g_strat, smp, bead_mask, res = s1.build_gating_strategy(
        gs, color_ranges, fcs, scatteronly
    )

    df = smp.as_dataframe(source="raw")
    fsc_max = df["fsc_a"].max()
    ssc_max = df["ssc_a"].max()

    _gs = gs.scatter_gates[fcs.filemeta.machine.om].from_color(fcs.filemeta.color)

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=bead_mask,
        x_max=min(_gs.fsc_max * 5, fsc_max),
        y_max=min(_gs.ssc_max * 5, ssc_max),
    )

    if scatteronly:
        return row(p0), res

    color_gates: dict[Color, list[tuple[float, float]]] = {}
    for gname, gpath in g_strat.get_child_gate_ids("beads", ("root",)):
        g = g_strat.get_gate(gname, gpath)
        # ASSUME each gate is a rectangle gate with one dimension
        color = g.get_dimension_ids()[0]
        _color = Color(color)
        dim = g.get_dimension(color)
        if _color not in color_gates:
            color_gates[_color] = []
        color_gates[_color].append((dim.min, dim.max))

    df_beads_x = smp.as_dataframe(source="xform", event_mask=bead_mask)
    smp_colors = Sample(df_beads_x, sample_id=str(fcs.indexed_path.filepath.name))
    ps = []
    for c in colors:
        p = smp_colors.plot_histogram(c.value, source="raw", x_range=(-0.2, 1))
        if c in color_gates:
            p.vspan(x=[g[0] for g in color_gates[c]], color="#ff0000")
            p.vspan(
                x=[g[1] for g in color_gates[c]], color="#00ff00", line_dash="dashed"
            )
        ps.append(p)

    return row(p0, *ps), res


def make_plots(
    boundaries: Path,
    files: Path,
    params: Path,
    om: OM,
    rainbow: bool,
    colors: list[Color],
    scatteronly: bool,
    debug: Path | bool,
) -> None:
    gating_config = ga.read_gates(boundaries).sop1
    calibrations = [c for c in s1.read_paths(files) if c.filemeta.machine.om == om]
    rangemap = ma.read_range_map(params)
    rainbow_cs, non_rainbow_cs = partition(
        lambda x: x.filemeta.color is None,
        calibrations,
    )

    non_rainbow_rows, non_rainbow_debug = unzip2(
        [
            apply_gates_to_sample(gating_config, rangemap, c, colors, scatteronly)
            for c in sorted(
                non_rainbow_cs,
                key=lambda x: (
                    10000 if x.filemeta.color is None else x.filemeta.color.index
                ),
            )
        ]
    )

    non_rainbow_tab = TabPanel(
        child=column(non_rainbow_rows),
        title="Non Rainbow",
    )
    rainbow_row, rainbow_debug = next(
        (
            apply_gates_to_sample(gating_config, rangemap, c, colors, scatteronly)
            for c in rainbow_cs
        ),
        (None, None),
    )
    if rainbow_row:
        rainbow_tab = TabPanel(child=column(rainbow_row), title="Rainbow")
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
    if debug is not None:
        with open(debug, "w") as f:
            arr = non_rainbow_debug + ([] if rainbow_debug is None else [rainbow_debug])
            json.dump([a.json for a in arr], f)
    show(page)


def list_oms(files: Path) -> None:
    xs = s1.read_path_map(files)
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
    plot_parser.add_argument("params", help="path to list of parameters")
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
    plot_parser.add_argument(
        "-s",
        "--scatteronly",
        action="store_true",
        help="only include scatter",
    )
    plot_parser.add_argument(
        "-d",
        "--debugpath",
        help="path to save debug json output",
    )

    # write_gate_parser = subparsers.add_parser(
    #     "write_gate",
    #     help="write gate for org/machine ID",
    # )
    # write_gate_parser.add_argument("om", help="org/machine ID")
    # write_gate_parser.add_argument("files", help="path to list of files")
    # write_gate_parser.add_argument("gates", help="path to list of gate ranges")
    # write_gate_parser.add_argument("params", help="path to list of params")
    # write_gate_parser.add_argument("-o", "--out", help="output directory")

    write_gates_parser = subparsers.add_parser("write_gates", help="write all gates")
    write_gates_parser.add_argument("files", help="path to list of files")
    write_gates_parser.add_argument("gates", help="path to list of gate ranges")
    write_gates_parser.add_argument("params", help="path to list of params")
    write_gates_parser.add_argument("-o", "--out", help="output directory")

    parsed = parser.parse_args(sys.argv[1:])

    if parsed.cmd == "list":
        list_oms(Path(parsed.files))

    if parsed.cmd == "plot":
        make_plots(
            Path(parsed.gates),
            Path(parsed.files),
            Path(parsed.params),
            OM(parsed.om),
            parsed.rainbow,
            [*Color] if parsed.colors is None else parsed.colors.split(","),
            parsed.scatteronly,
            parsed.debugpath,
        )

    # if parsed.cmd == "write_gate":
    #     s1.write_gate(
    #         s1.DEF_SC,
    #         s1.OM(parsed.om),
    #         Path(parsed.files),
    #         Path(parsed.gates),
    #         Path(parsed.params),
    #         fmap_maybe(Path, parsed.out),
    #     )

    if parsed.cmd == "write_gates":
        s1.write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
        )


main()

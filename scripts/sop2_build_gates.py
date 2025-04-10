#! /usr/bin/env python3

import argparse
import sys
from typing import Any
from tempfile import NamedTemporaryFile
from pathlib import Path
import flowkit as fk  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs
from common.functional import fmap_maybe
import common.sop2 as s2
import common.gating as ga
import common.metadata as ma


def apply_gates_to_sample(
    gs: ga.SOP2Gates,
    rm: ma.RangeMap,
    fcs: ma.FcsCompensationMeta,
    colors: list[ma.Color],
    scatteronly: bool,
    doublets: bool,
) -> Any:
    g_strat, smp, bead_mask = s2.build_gating_strategy(gs, rm, fcs, scatteronly)

    df = smp.as_dataframe(source="raw")
    fsc_max = df["fsc_a"].max()
    ssc_max = df["ssc_a"].max()

    _gs = (
        gs.scatter_gates[fcs.filemeta.machine.om]
        .from_material(fcs.filemeta.material)
        .from_color(fcs.filemeta.color)
    )

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=bead_mask,
        x_max=min(_gs.fsc_max * 5, fsc_max),
        y_max=min(_gs.ssc_max * 5, ssc_max),
    )

    if doublets:
        p1 = smp.plot_scatter(
            "fsc_a",
            "fsc_h",
            source="raw",
            highlight_mask=bead_mask,
        )
        ps = [p0, p1]
    else:
        ps = [p0]

    if scatteronly:
        return row(*ps)

    color_gates: dict[ma.Color, list[tuple[float, float]]] = {}
    for gname, gpath in g_strat.get_child_gate_ids("beads", ("root",)):
        g = g_strat.get_gate(gname, gpath)
        # ASSUME each gate is a rectangle gate with one dimension
        color = g.get_dimension_ids()[0]
        _color = ma.Color(color)
        dim = g.get_dimension(color)
        if _color not in color_gates:
            color_gates[_color] = []
        color_gates[_color].append((dim.min, dim.max))

    df_beads_x = smp.as_dataframe(source="xform", event_mask=bead_mask)
    smp_colors = fk.Sample(df_beads_x, sample_id=str(fcs.indexed_path.filepath.name))
    more_ps = []
    for c in colors:
        p = smp_colors.plot_histogram(c.value, source="raw", x_range=(-0.2, 1))
        if c in color_gates:
            p.vspan(x=[g[0] for g in color_gates[c]], color="#ff0000")
            p.vspan(
                x=[g[1] for g in color_gates[c]], color="#00ff00", line_dash="dashed"
            )
        more_ps.append(p)

    return row(*ps, *more_ps)


def make_plots(
    om: ma.OM,
    files: Path,
    gates: Path,
    params: Path,
    colors: list[ma.Color],
    matrices: list[ma.Matrix],
    scatteronly: bool,
    # debug: Path | bool,
) -> None:
    gating_config = ga.read_gates(gates).sop2
    comps = set([c for c in s2.read_paths(files) if c.filemeta.machine.om == om])
    mcomps = s2.group_by_matrix(comps)
    rm = ma.read_range_map(params)

    panels = [
        TabPanel(
            child=column(
                [
                    apply_gates_to_sample(
                        gating_config,
                        rm,
                        c,
                        colors,
                        scatteronly,
                        matrix is ma.Matrix.Matrix1,
                    )
                    for c in sorted(cs, key=lambda x: x.filemeta.color)
                ]
            ),
            title=matrix.name,
        )
        for matrix, cs in mcomps.items()
        if matrix in matrices
    ]

    # open but don't close temp file to save plot
    tf = NamedTemporaryFile(suffix="_sop1.html", delete_on_close=False, delete=False)
    output_file(tf.name)
    # if debug is not None:
    #     with open(debug, "w") as f:
    #         arr = non_rainbow_debug + ([] if rainbow_debug is None else [rainbow_debug])
    #         json.dump([a.json for a in arr], f)
    show(Tabs(tabs=panels))


def list_oms(files: Path) -> None:
    xs = sorted(list(set(x.filemeta.machine.om for x in s2.read_paths(files))))
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
        "-m",
        "--matrices",
        help="comma separated list of matrices to include (1, 2, or 3)",
    )
    # plot_parser.add_argument(
    #     "-d",
    #     "--debugpath",
    #     help="path to save debug json output",
    # )

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
            ma.OM(parsed.om),
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            (
                [*ma.Color]
                if parsed.colors is None
                else [ma.Color(c) for c in parsed.colors.split(",")]
            ),
            (
                [*ma.Matrix]
                if parsed.matrices is None
                else [ma.Matrix(int(m)) for m in parsed.matrices.split(",")]
            ),
            parsed.scatteronly,
            # parsed.debugpath,
        )

    if parsed.cmd == "write_gates":
        s2.write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
        )


main()

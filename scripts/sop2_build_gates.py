#! /usr/bin/env python3

import argparse
import numpy as np
import numpy.typing as npt
import sys
from itertools import groupby
from typing import Any, Iterator
from tempfile import NamedTemporaryFile
from pathlib import Path
import flowkit as fk  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs
from common.functional import fmap_maybe, partition
from common.io import read_fcs
import common.sop1 as s1
import common.gating as ga
import common.metadata as ma


PathMap = dict[ma.IndexedPath, tuple[ma.Color, ma.CompMaterial]]
MatrixPathMap = dict[ma.Matrix, PathMap]
OmMatrixPathMap = dict[ma.OM, MatrixPathMap]


def read_path_map(files: Path) -> OmMatrixPathMap:
    fs = ma.read_files(files)
    comps = [
        ma.SplitFCSMeta(f.indexed_path, f.filemeta)
        for f in fs
        if isinstance(f.filemeta, ma.CompensationMeta)
        and f.filemeta.machine.om not in ma.NO_SCATTER
    ]

    def expand_color(xs: list[ma.FcsCompensationMeta], more: PathMap) -> PathMap:
        ys = [
            *[(x.indexed_path, (x.filemeta.color, x.filemeta.material)) for x in xs],
            *more.items(),
        ]
        return dict(sorted(ys, key=lambda y: y[1][0]))

    def go(xs: Iterator[ma.FcsCompensationMeta]) -> MatrixPathMap:
        m1, rest1 = partition(lambda x: x.filemeta.matrix is ma.Matrix.Matrix1, xs)
        m4, rest4 = partition(lambda x: x.filemeta.matrix is None, rest1)
        m2, m3 = partition(lambda x: x.filemeta.matrix is ma.Matrix.Matrix2, rest4)
        _m4 = expand_color(m4, {})
        return {
            ma.Matrix.Matrix1: expand_color(m1, {}),
            ma.Matrix.Matrix2: expand_color(m2, _m4),
            ma.Matrix.Matrix3: expand_color(m3, _m4),
        }

    return {om: go(gs) for om, gs in groupby(comps, lambda c: c.filemeta.machine.om)}


def build_gating_strategy(
    conf: ga.SOP2Gates,
    gs: ga.AnyBounds,
    rm: ma.RangeMap,
    bead_color: ma.Color,
    fcs: ma.IndexedPath,
) -> tuple[fk.GatingStrategy, fk.Sample, npt.NDArray[np.bool_]]:
    g_strat = fk.GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    bead_gate = ga.build_scatter_gate(gs)
    g_strat.add_gate(bead_gate, ("root",))

    # The color gates are automatically placed according to events, so read
    # events, make a flowkit Sample, then gate out the beads
    parsed = read_fcs(fcs.filepath)
    smp = fk.Sample(parsed.events, sample_id=str(fcs.filepath.name))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    # if scatteronly:
    #     return g_strat, smp, mask, GatingStrategyDebug(fcs_path.name, {})

    # Apply logicle transform to each color channel. In this case there should
    # be relatively few events in the negative range, so A should be 0. Set M to
    # be 4.5 (sane default). I don't feel like getting the original range data
    # for each channel so just use the max of all of them (T = max). Then set W
    # according to 5% negative heuristic (see Parks et al (the Logicle Paper)
    # for formulas/rationale for doing this).
    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    trans = {}
    for c in ma.Color:
        arr = df_beads[c.value].values
        maxrange = float(rm[fcs.file_index][c])
        trans[c.value] = conf.transform_config.to_transform(arr, maxrange)

    for k, v in trans.items():
        g_strat.add_transform(f"{k}_logicle", v)

    smp.apply_transform(trans)
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    gate_results = {}
    # fc beads
    x = df_beads_x[bead_color.value].values
    r = ga.make_min_density_serial_gates(conf.autogate_config, x, 2)
    gate_results[bead_color.value] = r

    gates = [
        fk.gates.RectangleGate(
            f"{bead_color.value}_{i}",
            [
                fk.Dimension(
                    bead_color.value,
                    transformation_ref=f"{bead_color.value}_logicle",
                    range_min=s.x0,
                    range_max=s.x1,
                )
            ],
        )
        for i, s in enumerate(r.xintervals)
    ]
    for g in gates:
        g_strat.add_gate(g, ("root", "beads"))

    return g_strat, smp, mask


def apply_gates_to_sample(
    conf: ga.SOP2Gates,
    gs: ga.AnyBounds,
    bead_color: ma.Color,
    rm: ma.RangeMap,
    fcs: ma.IndexedPath,
    colors: list[ma.Color],
    scatteronly: bool,
    doublets: bool,
) -> Any:
    g_strat, smp, bead_mask = build_gating_strategy(conf, gs, rm, bead_color, fcs)

    df = smp.as_dataframe(source="raw")
    fsc_max = df["fsc_a"].max()
    ssc_max = df["ssc_a"].max()

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=bead_mask,
        x_max=min(gs.fsc_max * 5, fsc_max),
        y_max=min(gs.ssc_max * 5, ssc_max),
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
    smp_colors = fk.Sample(df_beads_x, sample_id=str(fcs.filepath.name))
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
    boundaries: Path,
    params: Path,
    colors: list[ma.Color],
    matrices: list[ma.Matrix],
    scatteronly: bool,
    # debug: Path | bool,
) -> None:
    gbs = ga.read_gates(boundaries).sop2
    gates = gbs.scatter_gates[om]
    path_map = read_path_map(files)[om]
    rm = ma.read_range_map(params)

    panels = [
        TabPanel(
            child=column(
                [
                    apply_gates_to_sample(
                        gbs,
                        gates.from_material(material).from_color(color),
                        color,
                        rm,
                        p,
                        colors,
                        scatteronly,
                        matrix is ma.Matrix.Matrix1,
                    )
                    for p, (color, material) in pmap.items()
                ]
            ),
            title=matrix.name,
        )
        for matrix, pmap in path_map.items()
        if matrix in matrices
    ]

    # # open but don't close temp file to save plot
    tf = NamedTemporaryFile(suffix="_sop1.html", delete_on_close=False, delete=False)
    output_file(tf.name)
    # if debug is not None:
    #     with open(debug, "w") as f:
    #         arr = non_rainbow_debug + ([] if rainbow_debug is None else [rainbow_debug])
    #         json.dump([a.json for a in arr], f)
    show(Tabs(tabs=panels))


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

    # TODO configure def sc
    if parsed.cmd == "plot":
        make_plots(
            # s1.DEF_SC,
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
        s1.write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
        )


main()

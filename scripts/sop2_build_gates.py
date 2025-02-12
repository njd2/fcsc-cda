#! /usr/bin/env python3

import math
import yaml  # type: ignore
import argparse
import numpy as np
import numpy.typing as npt
import sys

# from enum import Enum, unique
from itertools import groupby
from typing import Any, Iterator, NamedTuple, assert_never
from tempfile import NamedTemporaryFile
from pathlib import Path
import flowkit as fk  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs
from common.functional import fmap_maybe, partition
from pydantic import BaseModel as BaseModel_
from pydantic import Field
from common.io import read_fcs
import common.sop1 as s1
import common.metadata as ma

LogicleM = 4.5


# @unique
# class EID(Enum):
#     E1 = 1
#     E2 = 2
#     E3 = 3
#     E4 = 4

#     @classmethod
#     def from_matrix(cls, m: ma.Matrix | None) -> "EID":
#         if m is ma.Matrix.Matrix1:
#             return EID.E1
#         if m is ma.Matrix.Matrix2:
#             return EID.E2
#         if m is ma.Matrix.Matrix3:
#             return EID.E3
#         if m is None:
#             return EID.E4
#         else:
#             assert_never(m)


class BaseModel(BaseModel_):
    class Config:
        frozen = True
        extra = "forbid"


class GateInterval(NamedTuple):
    x0: float
    x1: float


class RectBounds(BaseModel):
    fsc: GateInterval
    ssc: GateInterval

    @property
    def fsc_min(self) -> float:
        return self.fsc.x0

    @property
    def fsc_max(self) -> float:
        return self.fsc.x1

    @property
    def ssc_min(self) -> float:
        return self.ssc.x0

    @property
    def ssc_max(self) -> float:
        return self.ssc.x1


class PolyPoint(NamedTuple):
    fsc: float
    ssc: float


class PolyBounds(BaseModel):
    vertices: list[PolyPoint] = Field(..., min_items=3)

    @property
    def fsc(self) -> list[float]:
        return [p.fsc for p in self.vertices]

    @property
    def ssc(self) -> list[float]:
        return [p.ssc for p in self.vertices]

    @property
    def fsc_min(self) -> float:
        return min(self.fsc)

    @property
    def fsc_max(self) -> float:
        return max(self.fsc)

    @property
    def ssc_min(self) -> float:
        return min(self.ssc)

    @property
    def ssc_max(self) -> float:
        return max(self.ssc)


AnyBounds = PolyBounds | RectBounds


class OverrideRectBounds(RectBounds):
    colors: list[ma.Color]


class OverridePolyBounds(PolyBounds):
    colors: list[ma.Color]


AnyOverrideBounds = OverridePolyBounds | OverrideRectBounds


class ExpGates(BaseModel):
    default: AnyBounds
    overrides: list[AnyOverrideBounds] = []

    def from_color(self, c: ma.Color) -> AnyBounds:
        try:
            return next((o for o in self.overrides if c in o.colors))
        except StopIteration:
            return self.default


class MachineGates(BaseModel):
    pbmc: ExpGates
    lyoleuk: ExpGates
    abc: ExpGates
    versa: ExpGates
    compbead: ExpGates

    def from_material(self, m: ma.CompMaterial) -> ExpGates:
        if m is ma.CompMaterial.PBMC:
            return self.pbmc
        elif m is ma.CompMaterial.LYOLEUK:
            return self.lyoleuk
        elif m is ma.CompMaterial.ABC:
            return self.abc
        elif m is ma.CompMaterial.VERSA:
            return self.versa
        elif m is ma.CompMaterial.COMP:
            return self.compbead
        else:
            assert_never(m)


class SOP2Gates(BaseModel):
    machines: dict[ma.OM, MachineGates]


def read_gates(p: Path) -> SOP2Gates:
    with open(p, "r") as f:
        return SOP2Gates.parse_obj(yaml.safe_load(f))


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
    gs: AnyBounds,
    rm: ma.RangeMap,
    bead_color: ma.Color,
    fcs: ma.IndexedPath,
) -> tuple[fk.GatingStrategy, fk.Sample, npt.NDArray[np.bool_]]:
    g_strat = fk.GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    if isinstance(gs, RectBounds):
        dim_fsc = fk.Dimension("fsc_a", range_min=gs.fsc.x0, range_max=gs.fsc.x1)
        dim_ssc = fk.Dimension("ssc_a", range_min=gs.ssc.x0, range_max=gs.ssc.x1)
        bead_gate = fk.gates.RectangleGate("beads", dimensions=[dim_fsc, dim_ssc])
    elif isinstance(gs, PolyBounds):
        dim_fsc = fk.Dimension("fsc_a")
        dim_ssc = fk.Dimension("ssc_a")
        bead_gate = fk.gates.PolygonGate(
            "beads",
            dimensions=[dim_fsc, dim_ssc],
            vertices=gs.vertices,
        )

    else:
        assert_never(gs)

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
    lt = fk.transforms.LogicleTransform
    for c in ma.Color:
        arr = df_beads[c.value].values
        arr_neg = arr[arr < 0]
        maxrange = float(rm[fcs.file_index][c])
        if arr_neg.size < 10:
            # TODO make these configurable
            trans[c.value] = lt(maxrange, 1.0, LogicleM, 0)
        else:
            low_ref = np.quantile(arr_neg, 0.05)
            # and these...
            best_W = (LogicleM - math.log10(maxrange / abs(low_ref))) / 2
            trans[c.value] = lt(maxrange, max(best_W, 0.25), LogicleM, 0)

    for k, v in trans.items():
        g_strat.add_transform(f"{k}_logicle", v)

    smp.apply_transform(trans)

    return g_strat, smp, mask


def apply_gates_to_sample(
    gs: AnyBounds,
    bead_color: ma.Color,
    rm: ma.RangeMap,
    fcs: ma.IndexedPath,
    colors: list[ma.Color],
    scatteronly: bool,
    doublets: bool,
) -> Any:
    g_strat, smp, bead_mask = build_gating_strategy(gs, rm, bead_color, fcs)

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

    # color_gates: dict[ma.Color, list[tuple[float, float]]] = {}
    # for gname, gpath in g_strat.get_child_gate_ids("beads", ("root",)):
    #     g = g_strat.get_gate(gname, gpath)
    #     # ASSUME each gate is a rectangle gate with one dimension
    #     color = g.get_dimension_ids()[0]
    #     _color = ma.Color(color)
    #     dim = g.get_dimension(color)
    #     if _color not in color_gates:
    #         color_gates[_color] = []
    #     color_gates[_color].append((dim.min, dim.max))

    if scatteronly:
        return row(*ps)

    df_beads_x = smp.as_dataframe(source="xform", event_mask=bead_mask)
    smp_colors = fk.Sample(df_beads_x, sample_id=str(fcs.filepath.name))
    more_ps = []
    for c in colors:
        p = smp_colors.plot_histogram(c.value, source="raw", x_range=(-0.2, 1))
        # if c in color_gates:
        #     p.vspan(x=[g[0] for g in color_gates[c]], color="#ff0000")
        #     p.vspan(
        #         x=[g[1] for g in color_gates[c]], color="#00ff00", line_dash="dashed"
        #     )
        more_ps.append(p)

    return row(*ps, *more_ps)


def make_plots(
    # sc: s1.SampleConfig,
    om: ma.OM,
    files: Path,
    boundaries: Path,
    params: Path,
    colors: list[ma.Color],
    matrices: list[ma.Matrix],
    scatteronly: bool,
    # debug: Path | bool,
) -> None:
    gbs = read_gates(boundaries).machines[om]
    path_map = read_path_map(files)[om]
    rm = ma.read_range_map(params)

    panels = [
        TabPanel(
            child=column(
                [
                    apply_gates_to_sample(
                        gbs.from_material(material).from_color(color),
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
            s1.DEF_SC,
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
        )


main()

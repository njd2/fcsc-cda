#! /usr/bin/env python3

import json
import math
import argparse
import sys
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from itertools import combinations
from tempfile import NamedTemporaryFile
from pathlib import Path
from functools import reduce
import flowkit as fk  # type: ignore
from flowkit import Dimension, Sample, GatingStrategy
from flowkit._models.gates import RectangleGate  # type: ignore
from flowkit._models.transforms import LogicleTransform  # type: ignore
from typing import Any, NamedTuple, NewType, TypeVar
from scipy.stats import gaussian_kde, norm  # type: ignore
from scipy.stats.mstats import mquantiles  # type: ignore
from scipy.integrate import trapezoid  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs
from common.io import read_fcs
from common.functional import fmap_maybe, from_maybe

X = TypeVar("X")

OM = NewType("OM", str)
Color = NewType("Color", str)
FCSName = NewType("FCSName", str)

RAINBOW_PAT = "RCP-30"

NO_SCATTER = [
    OM("LMNXSEA_CellStream-1"),
    OM("LMNXSEA_CellStream-2"),
    OM("LMNXSEA_ImageStreamX-1"),
]

COLORS = [*map(Color, ["v450", "v500", "fitc", "pc55", "pe", "pc7", "apc", "ac7"])]

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

COLOR_MAP = dict(zip(FILE_ORDER, COLORS))


LogicleM = 4.5


class GateRanges(NamedTuple):
    fsc_min: int
    fsc_max: int
    ssc_min: int
    ssc_max: int


# None Color = rainbow
GateRangeMap = dict[OM, dict[Color | None, GateRanges]]


class AutoGateConfig(NamedTuple):
    neighbor_frac: float
    min_prob: float
    bw: float | str
    inner_sigma: float
    outer_sigma: float


class SampleConfig(NamedTuple):
    non_rainbow: AutoGateConfig
    rainbow: AutoGateConfig


class D1Root(NamedTuple):
    """1st derivative roots of f, either a "peak" or "valley" for f."""

    x: float
    is_peak: bool


# TODO these are likely going to be flags, hardcoded for now to make things easy
DEF_SC = SampleConfig(
    non_rainbow=AutoGateConfig(
        bw=0.2,
        neighbor_frac=0.5,
        min_prob=0.1,
        inner_sigma=1,
        outer_sigma=1.96,
    ),
    rainbow=AutoGateConfig(
        bw=0.025,
        neighbor_frac=0.5,
        min_prob=0.08,
        inner_sigma=1,
        outer_sigma=1.96,
    ),
)


def path_to_color(p: Path) -> Color | None:
    return next((v for k, v in COLOR_MAP.items() if k in p.name), None)


V_F32 = npt.NDArray[np.float32]


def find_differential_peaks(x: V_F32, ddy: V_F32, is_peak: bool) -> list[D1Root]:
    signal = -2 if is_peak else 2
    # When we compute the double derivatives we consider the previous 2 points.
    # We care about the 2nd one which is either higher or lower than both the
    # points around it (ie peak or valley). The ddy array is missing the first 2
    # elements, so it actually starts counting from the 3rd point. Add 1 so the
    # resulting index lines up with the 2nd like we want.
    roots = np.where(ddy == signal)[0] + 1
    return [*map(lambda x: D1Root(x, is_peak), roots)]


def find_density_peaks(
    d: V_F32,
    bw_method: str | float,
    n: int = 512,
) -> tuple[V_F32, V_F32, list[int]]:
    """Find density peaks in vector x using "double diff" method.

    Density is computed with n evenly spaced intervals over x using a gaussian
    kernel.

    Specifically, find the places where the 1st derivative goes immediately from
    positive to negative (peaks) or the reverse (valleys).
    """

    if d.size < 3:
        raise ValueError("need at least two points to compute peaks/valleys")
    x = np.linspace(d.min(), d.max(), n)
    kernel = gaussian_kde(d, bw_method=bw_method)
    y = kernel(x)
    # TODO this will fail if the peak consists of more than one point (ie
    # multiple consecutive points are tied for the local maximum); in practice
    # these will be rare, but don't say I didn't warn you when this turns out
    # not to be forever ;)
    ddy = np.diff(np.sign(np.diff(y)))
    # peaks = find_differential_peaks(x, ddy, True)
    # valleys = find_differential_peaks(x, ddy, False)
    # return (x, y, sorted(peaks + valleys, key=lambda p: p.x))

    # When we compute the double derivatives we consider the previous 2 points.
    # We care about the 2nd one which is either higher or lower than both the
    # points around it (ie peak or valley). The ddy array is missing the first 2
    # elements, so it actually starts counting from the 3rd point. Add 1 so the
    # resulting index lines up with the 2nd like we want.
    valleys = (np.where(ddy == 2)[0] + 1).tolist()
    return (x, y, valleys)


def norm_quantiles(inner_sigma: float, outer_sigma: float) -> tuple[float, float]:
    if outer_sigma > inner_sigma > 0:
        ValueError("must be: outer > inner > 0")
    center = norm.cdf(0) - norm.cdf(-inner_sigma)
    tail = norm.cdf(-inner_sigma) - norm.cdf(-outer_sigma)
    q1 = tail / (center + tail) / 2
    return (q1, 1 - q1)


class NormalityTest(NamedTuple):
    left_passing: bool
    right_passing: bool
    xi: float
    x05: float
    x25: float
    x75: float
    x95: float
    xf: float
    dx: float

    @property
    def passing(self) -> bool:
        return self.right_passing and self.left_passing


class Interval(NamedTuple):
    a: int
    b: int


class TestInterval(NamedTuple):
    interval: Interval
    area: float
    normality: NormalityTest

    @property
    def json(self) -> dict[str, Any]:
        return {
            "interval": self.interval._asdict(),
            "area": self.area,
            "normality": self.normality._asdict(),
        }


IntervalTest = tuple[Interval, NormalityTest]


class XInterval(NamedTuple):
    x0: float
    x1: float


class ValleyPoint(NamedTuple):
    x: float
    y: float


class SerialGateResult(NamedTuple):
    # actual gates as list of intervals on the input data axis
    xintervals: list[XInterval]
    # debug stuff:
    # list of valley coordinates in the probability distribution
    valley_points: list[ValleyPoint]
    initial_intervals: list[TestInterval]
    merged_intervals: list[TestInterval]
    final_intervals: list[TestInterval]

    @property
    def json(self) -> dict[str, Any]:
        return {
            "xintervals": [i._asdict() for i in self.xintervals],
            "valley_points": [v._asdict() for v in self.valley_points],
            "initial_intervals": [i.json for i in self.initial_intervals],
            "merged_intervals": [i.json for i in self.merged_intervals],
            "final_intervals": [i.json for i in self.final_intervals],
        }


def slice_combinations(xs: list[Interval]) -> list[list[Interval]]:
    n = len(xs) - 1
    slices = [
        [c + 1 for c in cs] for i in range(n) for cs in combinations(range(n), i + 1)
    ]
    ys = [
        [Interval(xs[x].a, xs[y - 1].b) for x, y in zip([0, *ss], [*ss, len(xs)])]
        for ss in slices
    ]
    return [xs, *ys]


def make_min_density_serial_gates(
    ac: AutoGateConfig,
    d: V_F32,
    k: int,
) -> SerialGateResult:
    """Gate vector x by minimum density. Return at most k peaks"""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # x: discretized interval of vector d
    # y: KDE function values over x
    x, y, valleys = find_density_peaks(d, bw_method=ac.bw)
    valley_points = [ValleyPoint(float(x[v]), float(y[v])) for v in valleys]

    def compute_area(i: Interval) -> float:
        return float(trapezoid(y[i.a : i.b], x[i.a : i.b]))

    # Normality test overview:
    #
    # This test is designed to find peaks that are either "just noise" or those
    # that overlap sufficiently with their neighbor(s) that they are not totally
    # distinguishable.
    #
    # The basic assumption is that "good" peaks are normal, and that all gates
    # include some fraction of the total distribution. For sake of argument take
    # this fraction to be ~95%. In this case, the gate should be 4 sigma wide,
    # and ~72% of the data should be within 1 sigma (note this is different from
    # the usual 68% since we are normalizing to what is in the gate, which is
    # ~95%/2 sigma). We can easily test this by computing quantiles over the
    # data at 14%/86% and checking if they are 1 sigma away from either edge of
    # the gate. Stringency can be altered by changing the confidence interval,
    # which is equivalent to changing the fraction of data we expect to be
    # within the gate assuming normality.
    q1, q3 = norm_quantiles(ac.inner_sigma, ac.outer_sigma)

    def test_normality(i: Interval) -> NormalityTest:
        xi = float(x[i.a])
        xf = float(x[i.b])
        e = d[(xi <= d) & (d < xf)]
        res = mquantiles(e, (q1, q3))
        x25 = float(res[0])
        x75 = float(res[1])
        dx = 0.5 * (x75 - x25) * (ac.outer_sigma / ac.inner_sigma - 1)
        x05 = x25 - dx
        x95 = x75 + dx
        return NormalityTest(xi <= x05, x95 < xf, xi, x05, x25, x75, x95, xf, dx)

    # def merge_intervals(acc: list[IntervalTest], x: IntervalTest) -> list[IntervalTest]:
    #     # Overall rules:
    #     # - Intervals can only be merged if they are adjacent (end of previous
    #     #   is equal to start of current)
    #     # - Intervals are merged if either the previous or current interval
    #     #   fails the normality test.
    #     # - All final peaks must pass the normality test on both sides. This
    #     #   implies that we can drop non-adjacent intervals that fail the
    #     #   normality test.
    #     # - Intervals that are prior to the previous are considered OK and do
    #     #   not need to be considered.
    #     this_left = x[1].left_passing
    #     if len(acc) == 0:
    #         return [x] if this_left else []
    #     else:
    #         prev = acc[-1]
    #         prev_i = acc[-1][0]
    #         prev_right = prev[1].right_passing
    #         if prev_i.b == i.a:
    #             # If two intervals are adjacent and they are both normal, then
    #             # treat them as different peaks. Otherwise merge them. In the
    #             # next iteration we will check if this new peak should be
    #             # merged, added, or dropped.
    #             if prev_right and this_left:
    #                 rest = [prev, x]
    #             else:
    #                 merged_i = Interval(prev_i.a, i.b)
    #                 merged_test = test_normality(merged_i)
    #                 rest = [(merged_i, merged_test)]
    #         else:
    #             # If intervals are not adjacent, we cannot merge at all. Test
    #             # normality and drop either side that fails.
    #             rest = [prev] if prev_right else [] + [x] if this_left else []
    #         return acc[:-1] + rest

    # Zip valleys into intervals and compute probability in each interval,
    # removing intervals below threshold.

    def find_merge_combination(xs: list[Interval]) -> list[TestInterval]:
        combos = [
            [TestInterval(y, compute_area(y), test_normality(y)) for y in ys]
            for ys in slice_combinations(xs)
        ]
        return max(combos, key=lambda ys: sum((a for _, a, n in ys if n.passing)))

    intervals = [
        TestInterval(i := Interval(a, b), compute_area(i), test_normality(i))
        for a, b in zip([0, *valleys], [*valleys, x.size - 1])
    ]

    passing = [i.normality.passing and i.area >= ac.min_prob for i in intervals]
    passing_intervals = [i for i, t in zip(intervals, passing) if t]
    nonpassing_intervals = [i.interval for i, t in zip(intervals, passing) if not t]

    if len(nonpassing_intervals) > 0:
        grouped = reduce(
            lambda acc, i: (
                [*acc[:-1], [*acc[-1], i]] if acc[-1][-1].b == i.a else [*acc, [i]]
            ),
            nonpassing_intervals[1:],
            [[nonpassing_intervals[0]]],
        )
        merged_intervals = [
            c for g in grouped for c in find_merge_combination(g) if c.normality.passing
        ]
    else:
        merged_intervals = []

    final = [i for i in merged_intervals + passing_intervals if i.area > ac.min_prob]

    x_intervals = [
        XInterval(float(x[i.interval.a]), float(x[i.interval.b]))
        for i in sorted(sorted(final, key=lambda i: i.area)[:k])
    ]

    # TODO also return large peaks that failed so we know how much area we
    # missed and where it is
    return SerialGateResult(
        xintervals=x_intervals,
        valley_points=valley_points,
        initial_intervals=intervals,
        merged_intervals=merged_intervals,
        final_intervals=final,
    )

    # if len(big_intervals) > 0:
    #     # Merge/drop intervals depending on if they are adjacent and/or fail
    #     # the normality test.
    #     merged = reduce(merge_intervals, big_intervals[1:], [big_intervals[0]])
    #     # Test the last interval for normality on the right side, since this is
    #     # the only case that can't be caught with the reduction above.
    #     final = (
    #         merged if len(merged) > 0 and merged[-1][1].right_passing else merged[:-1]
    #     )
    # else:
    #     final = []

    # # Return top k intervals by area, sorted by position


class GatingStrategyDebug(NamedTuple):
    filename: str
    serial: dict[Color, SerialGateResult | None]

    @property
    def json(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "serial": {
                k: None if v is None else v.json for k, v in self.serial.items()
            },
        }


def build_gating_strategy(
    sc: SampleConfig,
    gs: GateRanges,
    color_ranges: dict[Color, int],
    fcs_path: Path,
    scatteronly: bool,
) -> tuple[GatingStrategy, Sample, npt.NDArray[np.bool_], GatingStrategyDebug]:
    bead_color = path_to_color(fcs_path)
    g_strat = GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    dim_fsc = Dimension("fsc_a", range_min=gs.fsc_min, range_max=gs.fsc_max)
    dim_ssc = Dimension("ssc_a", range_min=gs.ssc_min, range_max=gs.ssc_max)
    bead_gate = RectangleGate("beads", dimensions=[dim_fsc, dim_ssc])

    g_strat.add_gate(bead_gate, ("root",))

    # The color gates are automatically placed according to events, so read
    # events, make a flowkit Sample, then gate out the beads
    parsed = read_fcs(fcs_path)
    smp = Sample(parsed.events, sample_id=str(fcs_path.name))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    if scatteronly:
        return g_strat, smp, mask, GatingStrategyDebug(fcs_path.name, {})

    # Apply logicle transform to each color channel. In this case there should
    # be relatively few events in the negative range, so A should be 0. Set M to
    # be 4.5 (sane default). I don't feel like getting the original range data
    # for each channel so just use the max of all of them (T = max). Then set W
    # according to 5% negative heuristic (see Parks et al (the Logicle Paper)
    # for formulas/rationale for doing this).
    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    trans = {}
    for c in COLORS:
        arr = df_beads[c].values
        arr_neg = arr[arr < 0]
        maxrange = float(color_ranges[c])
        if arr_neg.size < 10:
            trans[c] = LogicleTransform(maxrange, 1.0, LogicleM, 0)
        else:
            low_ref = np.quantile(arr_neg, 0.05)
            best_W = (LogicleM - math.log10(maxrange / abs(low_ref))) / 2
            trans[c] = LogicleTransform(maxrange, best_W, LogicleM, 0)

    for k, v in trans.items():
        g_strat.add_transform(f"{k}_logicle", v)

    smp.apply_transform(trans)
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    # Place gates on each color channel. This will be different depending on if
    # these are rainbow beads or FC beads. The former should have 8 peaks in all
    # channels, and the latter should have exactly two peaks in its own color
    # channel. In all cases, place gate using peak/valley heuristic for finding
    # "large spikes" in the bead population. Do this on transformed data since
    # this is the only sane way to resolve the lower peaks.
    gate_results = {}
    for c in COLORS:
        # non rainbow beads should have two defined peaks in the channel for
        # that measures their color
        res = None
        x = df_beads_x[c].values
        if bead_color is not None and c == bead_color:
            res = make_min_density_serial_gates(sc.non_rainbow, x, 2)
        # rainbow beads are defined in all channels and there should be 8 peaks
        # at most
        elif bead_color is None:
            res = make_min_density_serial_gates(sc.rainbow, x, 8)
        gate_results[c] = res
        gates = (
            []
            if res is None
            else [
                RectangleGate(
                    f"{c}_{i}",
                    [
                        Dimension(
                            c,
                            transformation_ref=f"{c}_logicle",
                            range_min=s.x0,
                            range_max=s.x1,
                        )
                    ],
                )
                for i, s in enumerate(res.xintervals)
            ]
        )
        for g in gates:
            g_strat.add_gate(g, ("root", "beads"))
    return g_strat, smp, mask, GatingStrategyDebug(fcs_path.name, gate_results)


def apply_gates_to_sample(
    sc: SampleConfig,
    gs: GateRanges,
    color_ranges: dict[Color, int],
    fcs_path: Path,
    colors: list[Color],
    scatteronly: bool,
) -> tuple[Any, GatingStrategyDebug]:
    g_strat, smp, bead_mask, res = build_gating_strategy(
        sc, gs, color_ranges, fcs_path, scatteronly
    )

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
    smp_colors = Sample(df_beads_x, sample_id=str(fcs_path.name))
    ps = []
    for c in colors:
        p = smp_colors.plot_histogram(c, source="raw", x_range=(-0.2, 1))
        if c in color_gates:
            p.vspan(x=[g[0] for g in color_gates[c]], color="#ff0000")
            p.vspan(
                x=[g[1] for g in color_gates[c]], color="#00ff00", line_dash="dashed"
            )
        ps.append(p)

    return row(p0, *ps), res


FileMap = dict[OM, list[Path]]

RangeMap = dict[FCSName, dict[Color, int]]

FileRangeMap = dict[OM, dict[Path, dict[Color, int]]]


def read_path_map(files: Path) -> FileMap:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    ser = pd.read_table(files, usecols=[1], names=["file_path"])["file_path"]
    acc: FileMap = {}
    for _, file_path in ser.items():
        p = Path(file_path)
        xs = p.name.split("_")
        om = OM(f"{xs[2]}_{xs[3]}")
        if xs[5] == "SOP-01" and om not in NO_SCATTER:
            if om not in acc:
                acc[om] = []
            acc[om] += [p]
    for k in acc:
        acc[k].sort(
            key=lambda x: next(
                (i for i, o in enumerate(FILE_ORDER) if o in x.name), -1
            ),
        )
    return acc


def read_range_map(params: Path) -> RangeMap:
    df = pd.read_table(
        params,
        usecols=["filepath", "maxrange", "shortname"],
    )
    acc: RangeMap = {}
    for file_path, maxrange, shortname in df.itertuples(index=False):
        color = Color(shortname)
        if color in COLORS:
            p = FCSName(Path(file_path).name)
            if p not in acc:
                acc[p] = {}
            acc[p][color] = int(maxrange)
    return acc


def read_path_range_map(files: Path, params: Path) -> FileRangeMap:
    path_map = read_path_map(files)
    range_map = read_range_map(params)
    return {
        om: {p: range_map[FCSName(p.name)] for p in paths}
        for om, paths in path_map.items()
    }


def df_to_colormap(om: str, df: pd.DataFrame) -> dict[Color | None, GateRanges]:
    pairs = [
        (c, GateRanges(int(f0), int(f1), int(s0), int(s1)))
        for _, colors, f0, f1, s0, s1 in df.itertuples(index=False)
        for c in colors.split(",")
    ]

    all_colors = [p for p in pairs if p[0] == "all"]
    if len(all_colors) > 1:
        raise ValueError(f"got more than one 'all' for {om}")
    elif len(all_colors) == 1:
        all_range = all_colors[0][1]
    else:
        all_range = None

    rainbow_colors = [p for p in pairs if p[0] == "rainbow"]
    if len(rainbow_colors) != 1:
        raise ValueError(f"need exactly one rainbow color for {om}")
    else:
        rainbow_range = rainbow_colors[0][1]

    specific_colors = dict([p for p in pairs if p[0] not in ["all", "rainbow"]])

    if not all([p in COLORS for p in specific_colors]):
        raise ValueError(f"invalid colors for {om}")
    if all_range is None and len(specific_colors) != 8:
        raise ValueError(f"not all colors specified for {om}")

    if all_range is None:
        color_pairs = specific_colors
    else:
        xs = dict([(c, all_range) for c in COLORS if c not in specific_colors])
        color_pairs = {**xs, **specific_colors}

    return {**color_pairs, None: rainbow_range}


def read_gate_ranges(ranges_path: Path) -> GateRangeMap:
    return {
        OM(om[0]): df_to_colormap(om[0], df)
        for om, df in pd.read_table(ranges_path).groupby(["om"])
    }


def make_plots(
    ranges: Path,
    files: Path,
    params: Path,
    om: OM,
    rainbow: bool,
    colors: list[Color],
    scatteronly: bool,
    debug: Path | bool,
) -> None:
    all_gs = read_gate_ranges(ranges)
    path_range_map = read_path_range_map(files, params)
    paths = path_range_map[om]

    non_rainbow_results = [
        apply_gates_to_sample(
            DEF_SC,
            all_gs[om][path_to_color(p)],
            color_ranges,
            p,
            colors,
            scatteronly,
        )
        for p, color_ranges in paths.items()
        if RAINBOW_PAT not in p.name
    ]
    non_rainbow_rows = [r[0] for r in non_rainbow_results]
    non_rainbow_debug = [r[1] for r in non_rainbow_results]

    non_rainbow_tab = TabPanel(
        child=column(non_rainbow_rows),
        title="Non Rainbow",
    )
    rainbow_row, rainbow_debug = next(
        (
            apply_gates_to_sample(
                DEF_SC,
                all_gs[om][None],
                color_ranges,
                p,
                colors,
                scatteronly,
            )
            for p, color_ranges in paths.items()
            if RAINBOW_PAT in p.name
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


def write_gate_inner(
    om: OM,
    ranges: Path,
    color_ranges: dict[Color, int],
    fcs: Path,
    out: Path | None,
) -> None:
    all_gs = read_gate_ranges(ranges)
    color = path_to_color(fcs)
    gs, _, _, _ = build_gating_strategy(
        DEF_SC,
        all_gs[om][color],
        color_ranges,
        fcs,
        False,
    )
    if out is not None:
        with open(out, "wb") as f:
            fk.export_gatingml(gs, f)
    else:
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            fk.export_gatingml(gs, f)


def write_gate(
    om: OM,
    ranges: Path,
    params: Path,
    fcs: Path,
    out: Path | None,
) -> None:
    range_map = read_range_map(params)
    color_ranges = range_map[FCSName(fcs.name)]
    write_gate_inner(om, ranges, color_ranges, fcs, out)


def write_all_gates(
    files: Path,
    ranges: Path,
    params: Path,
    out_dir: Path | None,
    debug: bool,
) -> None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    path_range_map = read_path_range_map(files, params)
    for om, path_ranges in path_range_map.items():
        if debug:
            print(f"OM: {om}")
        for p, color_ranges in path_ranges.items():
            if debug:
                print(f"FCS file: {p.name}")
            color = from_maybe("rainbow", path_to_color(p))
            fn = f"{om}-{color}.xml"
            write_gate_inner(
                om, ranges, color_ranges, p, fmap_maybe(lambda p: p / fn, out_dir)
            )


def list_oms(files: Path) -> None:
    xs = read_path_map(files)
    for x in xs:
        print(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--verbose", "-v", action="count", default=0)

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

    write_gate_parser = subparsers.add_parser(
        "write_gate",
        help="write gate for org/machine ID",
    )
    write_gate_parser.add_argument("om", help="org/machine ID")
    write_gate_parser.add_argument("files", help="path to list of files")
    write_gate_parser.add_argument("gates", help="path to list of gate ranges")
    write_gate_parser.add_argument("params", help="path to list of params")
    write_gate_parser.add_argument("-o", "--out", help="output directory")

    write_gates_parser = subparsers.add_parser("write_gates", help="write all gates")
    write_gates_parser.add_argument("files", help="path to list of files")
    write_gates_parser.add_argument("gates", help="path to list of gate ranges")
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
            COLORS if parsed.colors is None else parsed.colors.split(","),
            parsed.scatteronly,
            parsed.debugpath,
        )

    if parsed.cmd == "write_gate":
        write_gate(
            OM(parsed.om),
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
        )

    if parsed.cmd == "write_gates":
        write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.params),
            fmap_maybe(Path, parsed.out),
            parsed.debug,
        )


main()

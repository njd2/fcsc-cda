import math
import sys
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from itertools import combinations, groupby
from pathlib import Path
from functools import reduce
import flowkit as fk  # type: ignore
from flowkit import Dimension, Sample, GatingStrategy
from flowkit._models.gates import RectangleGate  # type: ignore
from flowkit._models.transforms import LogicleTransform  # type: ignore
from typing import Any, NamedTuple, Generator
from scipy.stats import gaussian_kde  # type: ignore
from scipy.stats.mstats import mquantiles  # type: ignore
from scipy.integrate import trapezoid  # type: ignore
from multiprocessing import Pool
from common.io import read_fcs
from common.metadata import (
    OM,
    Color,
    read_files,
    CalibrationMeta,
    IndexedPath,
    FileIndex,
    split_path,
)
from common.functional import fmap_maybe, from_maybe, span, partition


# TODO need to deal with these things... :/
NO_SCATTER = [
    OM("LMNXSEA_CellStream-1"),
    OM("LMNXSEA_CellStream-2"),
    OM("LMNXSEA_ImageStreamX-1"),
]


LogicleM = 4.5


FileMap = dict[OM, list[IndexedPath]]
RangeMap = dict[FileIndex, dict[Color, int]]
FileRangeMap = dict[OM, dict[IndexedPath, dict[Color, int]]]
V_F32 = npt.NDArray[np.float32]


class GateBoundaries(NamedTuple):
    fsc_min: int
    fsc_max: int
    ssc_min: int
    ssc_max: int


# None = rainbow
GateRangeMap = dict[OM, dict[Color | None, GateBoundaries]]


class AutoGateConfig(NamedTuple):
    neighbor_frac: float
    min_prob: float
    bw: float | str
    tail_offset: float
    tail_prob: float


class SampleConfig(NamedTuple):
    non_rainbow: AutoGateConfig
    rainbow: AutoGateConfig


class D1Root(NamedTuple):
    """1st derivative roots of f, either a "peak" or "valley" for f."""

    x: float
    is_peak: bool


class ShapeTest(NamedTuple):
    xi: float
    x25: float
    x75: float
    xf: float
    dx: float

    @property
    def passing(self) -> bool:
        return self.right_passing and self.left_passing

    @property
    def left_passing(self) -> bool:
        return self.xi <= self.x05

    @property
    def right_passing(self) -> bool:
        return self.x95 < self.xf

    @property
    def x05(self) -> float:
        return self.x25 - self.dx

    @property
    def x95(self) -> float:
        return self.x75 + self.dx

    @property
    def json(self) -> dict[str, Any]:
        return {
            "left": self.left_passing,
            "right": self.right_passing,
            "xi": self.xi,
            "x05": self.x05,
            "x25": self.x25,
            "x75": self.x75,
            "x95": self.x95,
            "xf": self.xf,
            "dx": self.dx,
        }


class Interval(NamedTuple):
    a: int
    b: int


class TestInterval(NamedTuple):
    interval: Interval
    area: float
    shape: ShapeTest

    @property
    def json(self) -> dict[str, Any]:
        return {
            "interval": self.interval._asdict(),
            "area": self.area,
            "shape": self.shape.json,
        }


IntervalTest = tuple[Interval, ShapeTest]


class XInterval(NamedTuple):
    x0: float
    x1: float


class ValleyPoint(NamedTuple):
    x: float
    y: float


class SerialGateResult(NamedTuple):
    # actual gates as list of intervals on the input data axis
    xintervals: list[XInterval]

    # area under all the final peaks
    final_area: float

    # debug stuff:
    # list of valley coordinates in the probability distribution
    valley_points: list[ValleyPoint]
    # list of intervals generated from valleys
    initial_intervals: list[TestInterval]
    # list of intervals after the merge process
    merged_intervals: list[TestInterval]
    # list of all final intervals
    final_intervals: list[TestInterval]
    # list of intervals that failed the merge process
    failed_intervals: list[TestInterval]

    @property
    def json(self) -> dict[str, Any]:
        return {
            "xintervals": [i._asdict() for i in self.xintervals],
            "final_area": self.final_area,
            "valley_points": [v._asdict() for v in self.valley_points],
            "initial_intervals": [i.json for i in self.initial_intervals],
            "merged_intervals": [i.json for i in self.merged_intervals],
            "final_intervals": [i.json for i in self.final_intervals],
            "failed_intervals": [i.json for i in self.failed_intervals],
        }


class GatingStrategyDebug(NamedTuple):
    filename: str
    serial: dict[str, SerialGateResult]

    @property
    def json(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "serial": {
                k: None if v is None else v.json for k, v in self.serial.items()
            },
        }


# TODO these are likely going to be flags, hardcoded for now to make things easy
DEF_SC = SampleConfig(
    non_rainbow=AutoGateConfig(
        bw=0.2,
        neighbor_frac=0.5,
        min_prob=0.1,
        tail_offset=0.255,
        tail_prob=0.141,
    ),
    rainbow=AutoGateConfig(
        bw=0.025,
        neighbor_frac=0.5,
        min_prob=0.08,
        tail_offset=0.255 * 0.8,
        tail_prob=0.141,
    ),
)


def path_to_color(p: Path) -> Color | None:
    x = split_path(p).projection
    assert isinstance(x, CalibrationMeta)
    return x.color


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

    # When we compute the double derivatives we consider the previous 2 points.
    # We care about the 2nd one which is either higher or lower than both the
    # points around it (ie peak or valley). The ddy array is missing the first 2
    # elements, so it actually starts counting from the 3rd point. Add 1 so the
    # resulting index lines up with the 2nd like we want.
    valleys = (np.where(ddy == 2)[0] + 1).tolist()
    return (x, y, valleys)


def slice_combinations(xs: list[Interval]) -> Generator[list[Interval], None, None]:
    N = len(xs)
    n = N - 1
    if n < 0:
        yield []
    else:
        _as = [x.a for x in xs]
        _bs = [x.b for x in xs]
        yield [Interval(_as[0], _bs[-1])]
        for i in range(n):
            for cs in combinations(range(n), i + 1):
                ss = [c + 1 for c in cs]
                xys = zip([0, *ss], [*ss, N])
                yield [Interval(_as[x], _bs[y - 1]) for x, y in xys]


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

    def select_data(i: Interval) -> tuple[float, float, V_F32]:
        xi = float(x[i.a])
        xf = float(x[i.b])
        return xi, xf, d[(xi <= d) & (d < xf)]

    def is_empty(i: Interval) -> bool:
        return select_data(i)[2].size == 0

    def compute_area(i: Interval) -> float:
        float(trapezoid(y[i.a : i.b], x[i.a : i.b]))
        return float(trapezoid(y[i.a : i.b], x[i.a : i.b]))

    # Quantile test overview:
    #
    # This test is designed to find peaks that are either "just noise" or those
    # that overlap sufficiently with their neighbor(s) that they are not totally
    # distinguishable.
    #
    # We have two parameters that describe the key aspects of a
    # "good distribution". The first, tail_prob, defines that total probability
    # that should be in a "tail" (one sided). The second, tail_offset, describes
    # the length of the tail in data units (in this case whatever that means on
    # a logicle scale).
    #
    # The test involves finding the amount of data at the lower/upper quantiles
    # bordering each tail (defined by tail_prob), computing the distance between
    # these quantiles, then computing the allowed length of the tail based on
    # tail_offset. If either tail extends outside the gate boundaries, the test
    # fails. In other words, a bad distribution is one whose tails on either
    # side are too fat.
    #
    # This has a nice mapping to a normal distribution if one wants to be
    # mathematically purist. Setting tail_prob = 0.141 (probability b/t 1 and
    # 1.96) and tail_offset to 0.255 (1.96 sigma / 1 sigma / 2) corresponds to
    # expecting a gate to encapsulate at least 95% of a normal distribution.
    # This is nice in theory, but in practice many peaks are either skewed or
    # slightly fatter than normal, so one needs to use a bit of slack when
    # setting these values.

    q1 = ac.tail_prob
    q3 = 1 - ac.tail_prob
    f = ac.tail_offset / (1 - 2 * ac.tail_offset)

    def test_shape(i: Interval) -> ShapeTest:
        xi, xf, e = select_data(i)
        if e.size == 0:
            raise ValueError("attempting to get quantiles for empty vector")
        res = mquantiles(e, (q1, q3))
        x25 = float(res[0])
        x75 = float(res[1])
        return ShapeTest(xi, x25, x75, xf, (x75 - x25) * f)

    def find_merge_combination(xs: list[Interval]) -> list[TestInterval]:
        combos = [
            [TestInterval(y, compute_area(y), test_shape(y)) for y in ys]
            for ys in slice_combinations(xs)
        ]
        return max(combos, key=lambda ys: sum((a for _, a, n in ys if n.passing)))

    # Make intervals by interlacing valleys with 0 and N as initial/final bounds
    raw_intervals = [
        Interval(a, b) for a, b in zip([0, *valleys], [*valleys, x.size - 1])
    ]

    # Numpy will freak out if any intervals are empty when attempting to
    # compute quantiles. Merge empty intervals arbitrarily to deal with this.
    # Start by merging empty contiguous intervals on the left to the right,
    # and then merge empty intervals left going through the list.
    empty, rest = span(is_empty, raw_intervals)
    nonempty_left = (
        [Interval(empty[0].a, rest[0].b), *rest[1:]]
        if len(empty) > 0 and len(rest) > 0
        else rest
    )
    nonempty = reduce(
        lambda acc, i: (
            [*acc[:-1], Interval(acc[-1].a, i.b)] if is_empty(i) else acc + [i]
        ),
        nonempty_left[1:],
        [nonempty_left[0]],
    )

    # Compute the area and test the distribution shape for non-empty intervals.
    test_intervals = [TestInterval(i, compute_area(i), test_shape(i)) for i in nonempty]

    if len(test_intervals) > 0:
        _pass_nomerge, _fail = partition(
            lambda i: i.area >= ac.min_prob and i.shape.passing,
            test_intervals[1:],
        )
        # The first interval is unique because this is the only peak that can
        # plausibly be skewed so far to the left that it fails the left one side
        # of the shape test. This is because each distribution is fundamentally
        # a Poisson distribution (among others) and this will skew toward zero
        # at low-N, which is expected if the dynamic range and/or detector
        # voltage are low enough.
        first = test_intervals[0]
        pass_nomerge, fail = (
            ([first, *_pass_nomerge], _fail)
            if first.area >= ac.min_prob and first.shape.right_passing
            else (_pass_nomerge, [first, *_fail])
        )
    else:
        pass_nomerge = []
        fail = []

    # Merge combinations of failed intervals; keep the combinations that
    # maximizes area under valid peaks
    if len(fail) > 0:
        _fail_ints = [x.interval for x in fail]
        grouped = reduce(
            lambda acc, i: (
                [*acc[:-1], [*acc[-1], i]] if acc[-1][-1].b == i.a else [*acc, [i]]
            ),
            _fail_ints[1:],
            [[_fail_ints[0]]],
        )
        merged = [c for g in grouped for c in find_merge_combination(g)]
    else:
        merged = []

    # Combine everything and only keep the peaks above the area cutoff. Keep the
    # top k peaks and return final peaks sorted by position.
    pass_merge, fail_merge = partition(lambda i: i.shape.passing, merged)
    final = [i for i in pass_merge + pass_nomerge if i.area > ac.min_prob]
    final_k = sorted(final, key=lambda i: i.area)[:k]
    final_area = sum(i.area for i in final_k)
    x_intervals = sorted(
        XInterval(float(x[i.interval.a]), float(x[i.interval.b])) for i in final_k
    )

    return SerialGateResult(
        xintervals=x_intervals,
        valley_points=valley_points,
        initial_intervals=test_intervals,
        merged_intervals=pass_merge,
        final_intervals=final,
        failed_intervals=fail_merge,
        final_area=final_area,
    )


def build_gating_strategy(
    sc: SampleConfig,
    gs: GateBoundaries,
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
    for c in Color:
        arr = df_beads[c.value].values
        arr_neg = arr[arr < 0]
        maxrange = float(color_ranges[c])
        if arr_neg.size < 10:
            # TODO make these configurable
            trans[c.value] = LogicleTransform(maxrange, 1.0, LogicleM, 0)
        else:
            low_ref = np.quantile(arr_neg, 0.05)
            # and these...
            best_W = (LogicleM - math.log10(maxrange / abs(low_ref))) / 2
            trans[c.value] = LogicleTransform(maxrange, max(best_W, 0.25), LogicleM, 0)

    for k, v in trans.items():
        g_strat.add_transform(f"{k}_logicle", v)

    smp.apply_transform(trans)
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    # Place gates on each color channel. This will be different depending on if
    # these are rainbow beads or FC beads. The former should have 8 peaks in all
    # channels, and the latter should have exactly two peaks in its own color
    # channel. In the case of rainbow, only keep the gates for the color with
    # the most peaks (up to 8) and area under these peaks. In all cases, place
    # gate using peak/valley heuristic for finding "large spikes" in the bead
    # population. Do this on transformed data since this is the only sane way to
    # resolve the lower peaks.
    gate_results = {}
    if bead_color is None:
        # rainbow beads
        rs = [
            (
                c,
                make_min_density_serial_gates(
                    sc.rainbow,
                    df_beads_x[c.value].values,
                    8,
                ),
            )
            for c in Color
        ]
        # return all results in case we want to debug them...
        gate_results = {k.value: v for k, v in dict(rs).items()}
        # ...but only keep the best color in terms of peak resolution
        # TODO use some metric for "peak separation" here, like the distance b/t
        # the quantiles relative to the distances b/t gates
        maxres = max(rs, key=lambda r: (len(r[1].final_intervals), r[1].final_area))
        gate_color = maxres[0].value
        ints = maxres[1].xintervals
    else:
        # fc beads
        x = df_beads_x[bead_color.value].values
        r = make_min_density_serial_gates(sc.non_rainbow, x, 2)
        gate_results[bead_color.value] = r
        gate_color = bead_color.value
        ints = r.xintervals

    gates = [
        RectangleGate(
            f"{gate_color}_{i}",
            [
                Dimension(
                    gate_color,
                    transformation_ref=f"{gate_color}_logicle",
                    range_min=s.x0,
                    range_max=s.x1,
                )
            ],
        )
        for i, s in enumerate(ints)
    ]
    for g in gates:
        g_strat.add_gate(g, ("root", "beads"))
    return g_strat, smp, mask, GatingStrategyDebug(fcs_path.name, gate_results)


def read_path_map(files: Path) -> FileMap:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    fs = read_files(files)
    calibrations = [
        (f.indexed_path, f.filemeta)
        for f in fs
        if f.filemeta.machine.om not in NO_SCATTER
        and isinstance(f.filemeta, CalibrationMeta)
    ]
    return {
        om: [
            x[0]
            for x in sorted(
                gs, key=lambda x: 9 if x[1].color is None else x[1].color.index
            )
        ]
        for om, gs in groupby(calibrations, lambda c: c[1].machine.om)
    }

    # df = pd.read_table(files, names=["file_index", "file_path"])
    # acc: FileMap = {}
    # for file_index, file_path in df.itertuples(index=False):
    #     p = Path(file_path)
    #     xs = p.name.split("_")
    #     om = OM(f"{xs[2]}_{xs[3]}")
    #     if xs[5] == "SOP-01" and om not in NO_SCATTER:
    #         if om not in acc:
    #             acc[om] = []
    #         acc[om] += [FCSFile(FileIndex(int(file_index)), p)]
    # for k in acc:
    #     acc[k].sort(
    #         key=lambda x: next(
    #             (i for i, o in enumerate(FILE_ORDER) if o in x.path.name), -1
    #         ),
    #     )
    # return acc


def read_range_map(params: Path) -> RangeMap:
    df = pd.read_table(
        params,
        usecols=["file_index", "maxrange", "shortname"],
    )
    acc: RangeMap = {}
    for file_index, maxrange, shortname in df.itertuples(index=False):
        try:
            color = Color(shortname)
        except ValueError:
            color = None
        if color is not None:
            p = FileIndex(int(file_index))
            if p not in acc:
                acc[p] = {}
            acc[p][color] = int(maxrange)
    return acc


def read_path_range_map(files: Path, params: Path) -> FileRangeMap:
    path_map = read_path_map(files)
    range_map = read_range_map(params)
    return {
        om: {p: range_map[p.file_index] for p in paths}
        for om, paths in path_map.items()
    }


def df_to_colormap(om: str, df: pd.DataFrame) -> dict[Color | None, GateBoundaries]:
    pairs = [
        (c, GateBoundaries(int(f0), int(f1), int(s0), int(s1)))
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

    specific_colors: dict[Color | None, GateBoundaries] = {
        Color(p[0]): p[1] for p in pairs if p[0] not in ["all", "rainbow"]
    }

    if all_range is None and len(specific_colors) != 8:
        raise ValueError(f"not all colors specified for {om}")

    rest: dict[Color | None, GateBoundaries] = (
        {}
        if all_range is None
        else {c: all_range for c in Color if c not in specific_colors}
    )

    return {**specific_colors, **rest, None: rainbow_range}


def read_gate_ranges(ranges_path: Path) -> GateRangeMap:
    return {
        OM(om[0]): df_to_colormap(om[0], df)
        for om, df in pd.read_table(ranges_path).groupby(["om"])
    }


class GateRun(NamedTuple):
    sc: SampleConfig
    om: OM
    boundaries: Path
    color_ranges: dict[Color, int]
    fcs: IndexedPath
    out: Path | None


def write_gate_inner(r: GateRun) -> tuple[IndexedPath, Path | None]:
    all_gs = read_gate_ranges(r.boundaries)
    color = path_to_color(r.fcs.filepath)
    gs, _, _, _ = build_gating_strategy(
        r.sc,
        all_gs[r.om][color],
        r.color_ranges,
        r.fcs.filepath,
        False,
    )
    if r.out is not None:
        with open(r.out, "wb") as f:
            fk.export_gatingml(gs, f)
    else:
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            fk.export_gatingml(gs, f)
    return (r.fcs, r.out)


def write_gate(
    sc: SampleConfig,
    om: OM,
    boundaries: Path,
    params: Path,
    fcs: IndexedPath,
    out: Path | None,
) -> tuple[IndexedPath, Path | None]:
    range_map = read_range_map(params)
    color_ranges = range_map[fcs.file_index]
    return write_gate_inner(GateRun(sc, om, boundaries, color_ranges, fcs, out))


def write_all_gates(
    sc: SampleConfig,
    files: Path,
    boundaries: Path,
    params: Path,
    out_dir: Path | None,
    threads: int | None = None,
) -> list[tuple[IndexedPath, Path | None]]:
    def make_out_path(om: OM, p: Path) -> Path | None:
        color = from_maybe("rainbow", path_to_color(p))
        fn = f"{om}-{color}.xml"
        return fmap_maybe(lambda p: p / fn, out_dir)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    path_range_map = read_path_range_map(files, params)

    runs = [
        GateRun(
            sc,
            om,
            boundaries,
            color_ranges,
            p,
            make_out_path(om, p.filepath),
        )
        for om, path_ranges in path_range_map.items()
        for p, color_ranges in path_ranges.items()
    ]

    if threads is None:
        return list(map(write_gate_inner, runs))
    else:
        with Pool(threads) as pl:
            return pl.map(write_gate_inner, runs)

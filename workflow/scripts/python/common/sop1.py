import math
import sys
import os
import pandas as pd
import numpy as np
import numpy.typing as npt
from itertools import combinations
from pathlib import Path
from functools import reduce
import flowkit as fk  # type: ignore
from flowkit import Dimension, Sample, GatingStrategy
from flowkit._models.gates import RectangleGate  # type: ignore
from flowkit._models.transforms import LogicleTransform  # type: ignore
from typing import Any, NamedTuple, NewType
from scipy.stats import gaussian_kde  # type: ignore
from scipy.stats.mstats import mquantiles  # type: ignore
from scipy.integrate import trapezoid  # type: ignore
from common.io import read_fcs
from common.functional import fmap_maybe, from_maybe, span, partition

OM = NewType("OM", str)
Color = NewType("Color", str)
FCSName = NewType("FCSName", str)

FileMap = dict[OM, list[Path]]
RangeMap = dict[FCSName, dict[Color, int]]
FileRangeMap = dict[OM, dict[Path, dict[Color, int]]]
V_F32 = npt.NDArray[np.float32]

RAINBOW_PAT = "RCP-30"

# TODO need to deal with these things... :/
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
            "valley_points": [v._asdict() for v in self.valley_points],
            "initial_intervals": [i.json for i in self.initial_intervals],
            "merged_intervals": [i.json for i in self.merged_intervals],
            "final_intervals": [i.json for i in self.final_intervals],
            "failed_intervals": [i.json for i in self.failed_intervals],
        }


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
    return next((v for k, v in COLOR_MAP.items() if k in p.name), None)


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


def slice_combinations(xs: list[Interval]) -> list[list[Interval]]:
    n = len(xs) - 1
    if n < 0:
        return []
    slices = [
        [c + 1 for c in cs] for i in range(n) for cs in combinations(range(n), i + 1)
    ]
    ys = [
        [Interval(xs[x].a, xs[y - 1].b) for x, y in zip([0, *ss], [*ss, len(xs)])]
        for ss in slices
    ]
    return [[Interval(xs[0].a, xs[-1].b)], *ys]


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

    def test_normality(i: Interval) -> NormalityTest:
        xi, xf, e = select_data(i)
        if e.size == 0:
            raise ValueError("attempting to get quantiles for empty vector")
        res = mquantiles(e, (q1, q3))
        x25 = float(res[0])
        x75 = float(res[1])
        dx = (x75 - x25) * f
        x05 = x25 - dx
        x95 = x75 + dx
        return NormalityTest(xi <= x05, x95 < xf, xi, x05, x25, x75, x95, xf, dx)

    def find_merge_combination(xs: list[Interval]) -> list[TestInterval]:
        print(len(xs))
        combos = [
            [TestInterval(y, compute_area(y), test_normality(y)) for y in ys]
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

    # Compute the area and normality test for non-empty intervals
    test_intervals = [
        TestInterval(i, compute_area(i), test_normality(i)) for i in nonempty
    ]
    pass_nomerge, fail = partition(
        lambda i: i.area >= ac.min_prob and i.normality.passing,
        test_intervals,
    )

    # Merge combinations of failed intervals; keep the combinations that
    # maximizes area under valid peaks
    if len(fail) > 0:
        _fail = [x.interval for x in fail]
        grouped = reduce(
            lambda acc, i: (
                [*acc[:-1], [*acc[-1], i]] if acc[-1][-1].b == i.a else [*acc, [i]]
            ),
            _fail[1:],
            [[_fail[0]]],
        )
        merged = [c for g in grouped for c in find_merge_combination(g)]
    else:
        merged = []

    # Combine everything and only keep the peaks above the area cutoff. Keep the
    # top k peaks and return final peaks sorted by position.
    pass_merge, fail_merge = partition(lambda i: i.normality.passing, merged)
    final = [i for i in pass_merge + pass_nomerge if i.area > ac.min_prob]
    x_intervals = sorted(
        XInterval(float(x[i.interval.a]), float(x[i.interval.b]))
        for i in sorted(final, key=lambda i: i.area)[:k]
    )

    return SerialGateResult(
        xintervals=x_intervals,
        valley_points=valley_points,
        initial_intervals=test_intervals,
        merged_intervals=pass_merge,
        final_intervals=final,
        failed_intervals=fail_merge,
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


def write_gate_inner(
    sc: SampleConfig,
    om: OM,
    boundaries: Path,
    color_ranges: dict[Color, int],
    fcs: Path,
    out: Path | None,
) -> None:
    all_gs = read_gate_ranges(boundaries)
    color = path_to_color(fcs)
    gs, _, _, _ = build_gating_strategy(
        sc,
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
    sc: SampleConfig,
    om: OM,
    boundaries: Path,
    params: Path,
    fcs: Path,
    out: Path | None,
) -> None:
    range_map = read_range_map(params)
    color_ranges = range_map[FCSName(fcs.name)]
    write_gate_inner(sc, om, boundaries, color_ranges, fcs, out)


def write_all_gates(
    sc: SampleConfig,
    files: Path,
    boundaries: Path,
    params: Path,
    out_dir: Path | None,
) -> None:
    debug = False
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
                sc,
                om,
                boundaries,
                color_ranges,
                p,
                fmap_maybe(lambda p: p / fn, out_dir),
            )

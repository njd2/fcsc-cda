import math
import yaml  # type: ignore
import pandas as pd
from pathlib import Path
import numpy as np
from itertools import combinations
from functools import reduce
from typing import Any, NamedTuple, Generator, assert_never, Literal
import flowkit as fk  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from scipy.stats.mstats import mquantiles  # type: ignore
from scipy.integrate import trapezoid  # type: ignore
import common.metadata as ma
from common.functional import span, partition, fmap_maybe_def
from pydantic import BaseModel as BaseModel_
from pydantic import Field, validator, NonNegativeInt


V_F32 = np.ndarray[Literal[1], np.dtype[np.float32]]


class BaseModel(BaseModel_):
    class Config:
        frozen = True
        extra = "forbid"


class AutoGateConfig(BaseModel):
    bw: float | str = 0.2
    neighbor_frac: float = 0.5
    min_prob: float = 0.1
    tail_offset: float = 0.255
    tail_prob: float = 0.141


class GateInterval(NamedTuple):
    x0: float
    x1: float


class RectBounds(BaseModel):
    fsc: GateInterval
    ssc: GateInterval

    @validator("fsc", "ssc")
    def check_ranges(cls, v: GateInterval) -> GateInterval:
        assert v.x0 < v.x1, "gate must be positive"
        return v

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


class SOP1ScatterGates(BaseModel):
    fc: ExpGates
    rainbow: AnyBounds

    def from_color(self, c: ma.Color | None) -> AnyBounds:
        return fmap_maybe_def(self.rainbow, self.fc.from_color, c)


class SOP2ScatterGates(BaseModel):
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


class SOP1AutoGateConfig(BaseModel):
    fc: AutoGateConfig = AutoGateConfig(
        bw=0.2,
        neighbor_frac=0.5,
        min_prob=0.1,
        tail_offset=0.255,
        tail_prob=0.141,
    )
    rainbow: AutoGateConfig = AutoGateConfig(
        bw=0.025,
        neighbor_frac=0.5,
        min_prob=0.08,
        tail_offset=0.255 * 0.8,
        tail_prob=0.141,
    )


class LogicleConfig(BaseModel):
    # Number of decades (parameter "M"); this is usually a good default
    # according to original logicle paper
    m: float = 4.5

    # Cutoff for number of events beyond which the negative heuristic is used to
    # estimate W
    neg_cutoff: NonNegativeInt = 10

    # Parameter W for data with less than 'neg_cutoff' events below 0.
    nonneg_w: float = 1.0

    # Minimum parameter W for estimated W for data with more than 'neg_cutoff'
    # events below 0.
    neg_min_w: float = 0.25

    # The negative reference value to use when estimating W as a percentile.
    neg_percentile: float = 0.05

    def to_transform(self, x: V_F32, maxrange: float) -> fk.transforms.LogicleTransform:
        x_neg = x[x < 0]
        if x_neg.size < 10:
            w = self.nonneg_w
        else:
            low_ref = np.quantile(x_neg, self.neg_percentile)
            w = (self.m - math.log10(maxrange / abs(low_ref))) / 2
        return fk.transforms.LogicleTransform(maxrange, w, self.m, 0)


def transform_colors(
    lc: LogicleConfig,
    df: pd.DataFrame,
    ranges: dict[ma.Color, int],
) -> dict[ma.Color, tuple[str, fk.transforms.LogicleTransform]]:
    # ASSUME mostly safely that any data vector will be 32-bit floats
    trans = {
        c: (
            f"{c.value}_logicle",
            lc.to_transform(
                np.array(df[c.value].values, dtype=np.float32),
                float(ranges[c]),
            ),
        )
        for c in ma.Color
    }

    return trans

    # for k, v in trans.items():
    #     g_strat.add_transform(f"{k}_logicle", v)

    # smp.apply_transform(trans)
    # df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)


class SOP1TransformConfigs(BaseModel):
    fc: LogicleConfig = LogicleConfig()
    rainbow: LogicleConfig = LogicleConfig()


class SOP1Gates(BaseModel):
    autogate_configs: SOP1AutoGateConfig = SOP1AutoGateConfig()
    transform_configs: SOP1TransformConfigs = SOP1TransformConfigs()
    scatter_gates: dict[ma.OM, SOP1ScatterGates]

    # TODO validate the machine gate combos


class SOP2Gates(BaseModel):
    autogate_config: AutoGateConfig = AutoGateConfig(
        bw=0.2,
        neighbor_frac=0.5,
        min_prob=0.05,
        tail_offset=0.255,
        tail_prob=0.141,
    )
    transform_config: LogicleConfig = LogicleConfig()
    scatter_gates: dict[ma.OM, SOP2ScatterGates]

    # TODO validate the machine gate combos


class GateConfig(BaseModel):
    sop1: SOP1Gates
    sop2: SOP2Gates


def read_gates(p: Path) -> GateConfig:
    with open(p, "r") as f:
        return GateConfig.parse_obj(yaml.safe_load(f))


class XInterval(NamedTuple):
    x0: float
    x1: float


class ValleyPoint(NamedTuple):
    x: float
    y: float


class Interval(NamedTuple):
    a: int
    b: int


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


def build_scatter_gate(
    bounds: AnyBounds,
    name: str = "beads",
) -> fk.gates.PolygonGate | fk.gates.RectangleGate:
    if isinstance(bounds, RectBounds):
        dim_fsc = fk.Dimension(
            "fsc_a",
            range_min=bounds.fsc.x0,
            range_max=bounds.fsc.x1,
        )
        dim_ssc = fk.Dimension(
            "ssc_a",
            range_min=bounds.ssc.x0,
            range_max=bounds.ssc.x1,
        )
        return fk.gates.RectangleGate(name, dimensions=[dim_fsc, dim_ssc])
    elif isinstance(bounds, PolyBounds):
        return fk.gates.PolygonGate(
            name,
            dimensions=[fk.Dimension("fsc_a"), fk.Dimension("ssc_a")],
            vertices=bounds.vertices,
        )
    else:
        assert_never(bounds)

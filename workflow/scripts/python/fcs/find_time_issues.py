import gzip
import math
import numpy.typing as npt
import numpy as np
import pandas as pd
import scipy.optimize as spo  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any, TextIO, NewType, TypeVar, Generic
from dataclasses import dataclass, astuple
from multiprocessing import Pool
from common.io import read_fcs

X = TypeVar("X")

MinEvents = NewType("MinEvents", int)
I0 = NewType("I0", int)
I1 = NewType("I1", int)

Anomalies = npt.NDArray[np.uint8]
TimeValues = npt.NDArray[np.float32]


class MinEventsSOP(NamedTuple):
    sop1: MinEvents
    sop2: MinEvents
    sop3: MinEvents


# Reader Monad ;)
class Params(NamedTuple):
    min_events: MinEventsSOP
    spike_limit: float
    gap_limit: float
    non_mono_limit: float
    min_size: int
    rate_thresh: float


@dataclass(frozen=True)
class Interval(Generic[X]):
    """Interval on some dimension.

    By convention, x in within interval iif x0 <= x < x1 (ie slice notation).
    """

    x0: X
    x1: X


EventInterval = Interval[int]
TimeInterval = Interval[float]


def events_to_time(i: EventInterval, t: TimeValues) -> TimeInterval:
    return Interval(t[i.x0], t[i.x1])


class _Gate:
    def __init__(self, i0: int, i1: int, n: MinEvents, t: TimeValues):
        i = Interval(i0, i1)
        self.event: EventInterval = i
        self.time: TimeInterval = events_to_time(i, t)
        self.min_events = n

    @property
    def _serial(self) -> tuple[int, int, float, float, bool]:
        return (*astuple(self.event), *astuple(self.time), self.valid)

    @property
    def valid(self) -> bool:
        return self.n_events >= self.min_events

    @property
    def n_events(self) -> int:
        return self.event.x1 - self.event.x0 + 1

    def __repr__(self) -> str:
        return (
            f"Gate(interval=({self.event.x0}, {self.event.x1}), min={self.min_events})"
        )


class FlatGate(_Gate):
    def __init__(
        self,
        i0: int,
        i1: int,
        n: MinEvents,
        t: TimeValues,
        parent_anomaly: int,
        flat_break: bool,
    ):
        super().__init__(i0, i1, n, t)
        self.parent_anomaly = parent_anomaly
        self.flat_break = flat_break

    @property
    def serial(self) -> tuple[int, int, float, float, bool, int, bool]:
        return (*super()._serial, self.parent_anomaly, self.flat_break)


class AnomalyGate(_Gate):
    def __init__(
        self,
        i0: int,
        i1: int,
        n: MinEvents,
        t: TimeValues,
        anomaly: int,
    ):
        super().__init__(i0, i1, n, t)
        self.anomaly = anomaly

    @property
    def serial(self) -> tuple[int, int, float, float, bool, int]:
        return (*super()._serial, int(self.anomaly))


class RunConfig(NamedTuple):
    params: Params
    file_index: int
    path: Path
    sop: int


class GateResult(NamedTuple):
    anomaly_gates: list[AnomalyGate]
    flat_gates: list[FlatGate]
    time: TimeValues | None


class RunResult(NamedTuple):
    file_index: int
    result: GateResult


def get_min_events(p: Params, sop: int) -> MinEvents:
    m = p.min_events
    if sop == 1:
        return m.sop1
    elif sop == 2:
        return m.sop2
    elif sop == 3:
        return m.sop3
    else:
        # TODO make this cleaner
        raise ValueError("Invalid SOP")


def get_anomalies(p: Params, t: TimeValues) -> Anomalies:
    tdiff = t[1:] - t[:-1]
    tdiff_norm = tdiff / (t.max() - t.min())

    neg_spike = np.zeros((t.size), dtype=bool)
    large_gap = np.zeros((t.size), dtype=bool)
    non_mono = np.zeros((t.size), dtype=bool)

    # TODO this is confusing because it assumes that spike limit is bigger than
    # gap limit and non-mono limit.

    # Large negative spikes occur when the current point is less than the
    # previous point by some large margin, and the point after comes back up by
    # the same margin or greater
    neg_spike[1:-1] = (tdiff[:-1] < -p.spike_limit) & (tdiff[1:] > p.spike_limit)
    # Large gaps occur when the previous point is much less than the current
    # point by some large margin. Since negative spikes will by definition
    # produce large increases, mask out the points for which a negative spike
    # occurred immediately previous
    large_gap[2:] = tdiff_norm[1:] > p.gap_limit
    large_gap[2:] = ~neg_spike[1:-1] & large_gap[2:]
    # Non monotonic events occur when the current point is less than the
    # previous point by some margin. Assume this gets overridden below if it is
    # also a spike.
    non_mono[1:-1] = tdiff[:-1] < -p.non_mono_limit

    return np.select(
        [neg_spike, large_gap, non_mono],
        [1, 2, 3],
        default=0,
    ).astype(np.uint8)


def get_anomaly_gates(
    anomalies: npt.NDArray[np.uint8],
    n: MinEvents,
    t: TimeValues,
) -> list[AnomalyGate]:
    total = anomalies.size
    if anomalies.sum() == 0:
        return [AnomalyGate(0, total - 1, n, t, 0)]
    else:
        mask = (anomalies == 2) | (anomalies == 3)
        anomaly_positions = np.where(mask)[0].tolist()
        starts: list[int] = [0, *anomaly_positions]
        ends: list[int] = [*anomaly_positions, total - 1]
        codes = [*anomalies[mask].tolist(), 0]
        return [AnomalyGate(s, e, n, t, c) for s, e, c in zip(starts, ends, codes)]


def ss(xs: TimeValues) -> float:
    x: float = np.square(xs - xs.mean()).sum()
    return x


def r2(gate: float, i0: int, i1: int, xs: TimeValues, sst: float) -> float:
    g0 = int(gate)
    ss0 = ss(xs[i0 : (g0 - 1)]) if g0 - 1 > i0 else 0
    ss1 = ss(xs[g0 : i1 - 1]) if i1 - 1 > g0 else 0
    return 1 - (1 - (ss0 + ss1) / sst)


def find_flat(
    p: Params,
    acc: list[int],
    i0: int,
    i1: int,
    tdiff: TimeValues,
) -> list[int]:
    # This function is a basic step function curve fitter. The "gate" is the
    # boundary of the step, and it is placed such that the means of the two
    # partitions on either side of the step result in a maximized R^2, where R^2
    # is 1 - (SS1 + SS2) / SStot where SS1/2 are the sum of squares for each
    # partition and SStot is the sum of squares for the entire vector. This
    # function will continue partitioning the vector until an arbitrary
    # threshold is achieved
    #
    # NOTE each of these slices has a -1 for the top of the interval since by
    # convention the "gates" in this code are [i0, i1)
    sst = ss(tdiff[i0 : i1 - 1])
    # set tolerance to 0.5 so we stop the model at the nearest integer-ish
    res = spo.minimize_scalar(
        r2,
        method="bounded",
        options={"xatol": 0.5},
        args=(i0, i1, tdiff, sst),
        bounds=(i0, i1),
    )
    gate = int(res.x)
    rate_diff = math.log10(tdiff[i0 : gate - 1].mean() / tdiff[gate : i1 - 1].mean())
    # if gate is larger than our minimum size and produces two partitions with
    # flow rates that differ beyond our threshold, place the gate, and try to
    # gate the two new partitions, otherwise return whatever gates we have so
    # far, which may be none
    if (
        abs(rate_diff) > p.rate_thresh
        and i0 + p.min_size <= gate
        and gate <= i1 - p.min_size
    ):
        acc = find_flat(p, acc, i0, gate, tdiff)
        acc = [*acc, gate]
        acc = find_flat(p, acc, gate + 1, i1, tdiff)
        return acc
    else:
        return acc


def get_flat_gates(
    p: Params,
    ag: AnomalyGate,
    tdiff: TimeValues,
    t: TimeValues,
    n: MinEvents,
) -> list[FlatGate]:
    # NOTE add 1 since the first diff value for this gate is not differentiable
    gates = find_flat(p, [], ag.event.x0 + 1, ag.event.x1, tdiff)
    if len(gates) == 0 or not ag.valid:
        return [FlatGate(ag.event.x0, ag.event.x1, n, t, ag.anomaly, False)]
    else:
        starts = [ag.event.x0, *gates]
        ends = [*gates, ag.event.x1]
        return [
            FlatGate(s, e, n, t, ag.anomaly, s > ag.event.x0)
            for s, e in zip(starts, ends)
        ]


def get_all_gates(c: RunConfig) -> RunResult:
    parsed = read_fcs(c.path)

    min_events = get_min_events(c.params, c.sop)
    t = parsed.events["time"].values.astype(np.float32)
    anomalies = get_anomalies(c.params, t)

    # TODO this can be massively simplified by only exporting flat gates
    ano_gates = get_anomaly_gates(anomalies, min_events, t)
    n_ano_valid = sum(g.valid for g in ano_gates)

    if n_ano_valid == 0:
        res = GateResult(ano_gates, [], t)
    else:
        t_clean = t[anomalies != 1]
        tdiff_clean = t_clean[1:] - t_clean[:-1]
        flat_gates = [
            fg
            for ag in ano_gates
            for fg in get_flat_gates(c.params, ag, tdiff_clean, t, min_events)
        ]
        n_flat_valid = sum(g.valid for g in flat_gates)
        n_total = len(flat_gates) + len(ano_gates) - n_ano_valid
        res = GateResult(
            ano_gates,
            flat_gates,
            # include time vector if we have no valid flat gates and we have at
            # least one gate, in which case we want to plot the gates to inspect
            t if n_flat_valid == 0 or n_total > 1 else None,
        )
    return RunResult(c.file_index, res)


def main(smk: Any) -> None:
    sp = smk.params
    me = sp["min_events"]
    tl = sp["time_limits"]
    params = Params(
        min_events=MinEventsSOP(
            sop1=me["sop1"],
            sop2=me["sop2"],
            sop3=me["sop3"],
        ),
        spike_limit=tl["spike_limit"],
        gap_limit=tl["gap_limit"],
        non_mono_limit=tl["non_mono_limit"],
        rate_thresh=math.log10(100 / tl["rate_thresh"]),
        min_size=tl["min_size"],
    )
    meta_in = Path(smk.input["meta"])
    files_in = Path(smk.input["files"])

    anomaly_out = Path(smk.output["anomaly"])
    flat_out = Path(smk.output["flat"])
    top_out = Path(smk.output["top"])
    event_out = Path(smk.output["events"])

    META_COLUMNS = ["file_index", "sop"]

    df_files = pd.read_table(
        files_in,
        names=["file_index", "filepath"],
    ).set_index("file_index")
    df_meta = pd.read_table(meta_in)[META_COLUMNS].set_index("file_index")
    df = df_files.join(df_meta)
    runs = [
        RunConfig(params, int(file_index), Path(filepath), int(sop))
        for file_index, filepath, sop in df.itertuples(index=True)
    ]

    # weeeeeeeee
    with Pool(smk.threads) as pl:
        gate_results = pl.map(get_all_gates, runs)
    # gate_results = map(get_all_gates, runs[1476:1477])
    # gate_results = map(get_all_gates, runs[1483:1484])

    with (
        gzip.open(anomaly_out, "wt") as ao,
        gzip.open(flat_out, "wt") as fo,
        gzip.open(top_out, "wt") as to,
        gzip.open(event_out, "wt") as eo,
    ):
        for r in gate_results:
            fi = str(int(r.file_index))
            res = r.result

            def write_tsv_line(b: TextIO, xs: list[Any]) -> None:
                b.write("\t".join([fi, *[str(x) for x in xs]]) + "\n")

            for ag in res.anomaly_gates:
                write_tsv_line(ao, [*ag.serial])
            for fg in res.flat_gates:
                write_tsv_line(fo, [*fg.serial])
            # last gate in the flat series is the longest, so use that one
            g0 = next(
                iter(
                    sorted(
                        (g for g in res.flat_gates if g.valid),
                        key=lambda g: -g.n_events,
                    )
                ),
                None,
            )
            if g0 is not None:
                write_tsv_line(to, [*g0.serial])
            if res.time is not None:
                for ei in range(res.time.size):
                    write_tsv_line(eo, [ei, res.time[ei]])


main(snakemake)  # type: ignore

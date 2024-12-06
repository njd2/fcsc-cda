import gzip
import math
import numpy.typing as npt
import numpy as np
import pandas as pd
import scipy.optimize as spo  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any, TextIO, NewType
from dataclasses import dataclass
from multiprocessing import Pool
from common.io import read_fcs

ChannelMap = dict[tuple[str, str], str]
MinEvents = NewType("MinEvents", int)
I0 = NewType("I0", int)
I1 = NewType("I1", int)


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
class Gate:
    start: I0
    end: I1
    min_events: MinEvents

    @property
    def valid(self) -> int:
        return self.length >= self.min_events

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class AnomalyGate(Gate):
    anomaly: int | None


class RunConfig(NamedTuple):
    params: Params
    file_index: int
    path: Path
    channel: str
    sop: int


class GateResult(NamedTuple):
    anomaly_gates: list[AnomalyGate]
    flat_gates: list[Gate]


class ErrorResult(NamedTuple):
    time: "pd.Series[float]"
    ano_gates: list[AnomalyGate]
    flat_gates: list[Gate]


class RunResult(NamedTuple):
    file_index: int
    result: GateResult | ErrorResult


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


def read_time_channel_mapping(p: Path) -> ChannelMap:
    with open(p, "r") as f:
        next(f, None)
        return {
            (s[0], s[1]): s[2] for x in f if (s := x.rstrip().split("\t"))[3] == "time"
        }


def get_anomalies(p: Params, t: "pd.Series[float]") -> npt.NDArray[np.int64]:
    tdiff = t - t.shift(1)
    tdiff_norm = tdiff / (t.max() - t.min())

    large_neg = (tdiff < -p.spike_limit) & (tdiff.shift(-1) > p.spike_limit)
    neg_mask = ~large_neg.shift(1, fill_value=False)
    large_gap = (tdiff_norm > p.gap_limit) & neg_mask
    non_mono = tdiff < -p.non_mono_limit

    return np.select(
        [large_neg, large_gap, non_mono],
        [1, 2, 3],
        default=0,
    )


def get_anomaly_gates(
    anomalies: npt.NDArray[np.int64],
    n: MinEvents,
) -> list[AnomalyGate]:
    total = anomalies.size
    if anomalies.sum() == 0:
        return [AnomalyGate(I0(0), I1(total - 1), n, None)]
    else:
        mask = anomalies > 1
        anomaly_positions = np.where(mask)[0].tolist()
        starts: list[int] = [0, *anomaly_positions]
        ends: list[int] = [*anomaly_positions, total - 1]
        codes: list[int | None] = [*anomalies[mask].tolist(), None]
        return [AnomalyGate(I0(s), I1(e), n, c) for s, e, c in zip(starts, ends, codes)]


def ss(xs: "pd.Series[float]") -> float:
    x: float = np.square(xs - xs.mean()).sum()
    return x


def r2(gate: float, i0: int, i1: int, xs: "pd.Series[float]", sst: float) -> float:
    g0 = int(gate)
    ss0 = ss(xs[i0:g0])
    ss1 = ss(xs[g0 + 1 : i1])
    return 1 - (1 - (ss0 + ss1) / sst)


def find_flat(
    p: Params, acc: list[int], i0: int, i1: int, tdiff: "pd.Series[float]"
) -> list[int]:
    # This function is a basic step function curve fitter. The "gate" is the
    # boundary of the step, and it is placed such that the means of the two
    # partitions on either side of the step result in a maximized R^2, where R^2
    # is 1 - (SS1 + SS2) / SStot where SS1/2 are the sum of squares for each
    # partition and SStot is the sum of squares for the entire vector. This
    # function will continue partitioning the vector until an arbitrary
    # threshold is achieved
    #
    # NOTE thresh should be a positive number
    sst = ss(tdiff[i0:i1])
    # set tolerance to 0.5 so we stop the model at the nearest integer-ish
    res = spo.minimize_scalar(
        r2,
        method="bounded",
        options={"xatol": 0.5},
        args=(i0, i1, tdiff, sst),
        bounds=(i0, i1),
    )
    gate = i0 + int(res.x)
    rate_diff = math.log10(tdiff[i0:gate].mean() / tdiff[(gate + 1) : i1].mean())
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
    i0: I0,
    i1: I1,
    tdiff: "pd.Series[float]",
    n: MinEvents,
) -> list[Gate]:
    # NOTE add 1 since the first diff value for this gate is not differentiable
    gates = find_flat(p, [], i0 + 1, i1, tdiff)
    if len(gates) == 0:
        return [Gate(i0, i1, n)]
    else:
        starts = [i0, *gates]
        ends = [*gates, i1]
        return [Gate(I0(s), I1(e), n) for s, e in zip(starts, ends)]


def get_all_gates(c: RunConfig) -> RunResult:
    parsed = read_fcs(c.path)

    min_events = get_min_events(c.params, c.sop)
    t = parsed.events[c.channel]
    anomalies = get_anomalies(c.params, t)

    ano_gates = get_anomaly_gates(anomalies, min_events)
    # valid_ano_gates = [g for g in ano_gates if g.valid]

    if sum(g.valid for g in ano_gates) == 0:
        res: GateResult | ErrorResult = ErrorResult(t, ano_gates, [])
    else:
        t_clean = t[anomalies == 0]
        tdiff_clean = t_clean - t_clean.shift(1)
        flat_gates = [
            fg
            for ag in ano_gates
            if ag.valid
            for fg in get_flat_gates(
                c.params, ag.start, ag.end, tdiff_clean, min_events
            )
        ]
        if sum(g.valid for g in flat_gates) == 0:
            res = ErrorResult(t, ano_gates, flat_gates)
        else:
            res = GateResult(ano_gates, flat_gates)
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
    channel_in = Path(smk.input["channels"])
    meta_in = Path(smk.input["meta"])

    anomaly_out = Path(smk.output["anomaly"])
    flat_out = Path(smk.output["flat"])
    top_out = Path(smk.output["top"])
    event_out = Path(smk.output["events"])

    channels = read_time_channel_mapping(channel_in)

    META_COLUMNS = [
        "file_index",
        "org",
        "machine",
        "sop",
        "filepath",
        "missing_time",
        "percent_complete",
    ]

    df = pd.read_table(meta_in)[META_COLUMNS]
    runs = [
        RunConfig(params, i, Path(p), channels[(o, m)], s)
        for i, o, m, s, p, no_time, complete in df.itertuples(index=False)
        if i is not None and no_time is False and complete >= 100
    ]

    # weeeeeeeee
    with Pool(smk.threads) as pl:
        gate_results = pl.map(get_all_gates, runs)

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

            if isinstance(res, GateResult):
                for ag in res.anomaly_gates:
                    write_tsv_line(ao, [ag.start, ag.end, ag.anomaly, ag.valid])
                for fg in res.flat_gates:
                    write_tsv_line(fo, [fg.start, fg.end, fg.valid])
                # last gate in the flat series is the longest, so use that one
                g0 = next(
                    iter(
                        sorted(
                            (g for g in res.flat_gates if g.valid),
                            key=lambda g: -g.length,
                        )
                    ),
                    None,
                )
                if g0 is not None:
                    write_tsv_line(to, [g0.start, g0.end])
            else:
                for ag in res.ano_gates:
                    write_tsv_line(ao, [ag.start, ag.end, ag.anomaly, ag.valid])
                for fg in res.flat_gates:
                    write_tsv_line(fo, [fg.start, fg.end, fg.valid])
                for ei, t in res.time.items():
                    write_tsv_line(eo, [ei, t])


main(snakemake)  # type: ignore

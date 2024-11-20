import gzip
import math
import warnings
import numpy.typing as npt
import numpy as np
import pandas as pd
import fcsparser as fp  # type: ignore
import scipy.optimize as spo  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any, TextIO
from dataclasses import dataclass, asdict
from multiprocessing import Pool

ChannelMap = dict[tuple[str, str], str]


class MinEvents(NamedTuple):
    sop1: int
    sop2: int
    sop3: int


# Reader Monad ;)
class Params(NamedTuple):
    min_events: MinEvents
    spike_limit: float
    gap_limit: float
    non_mono_limit: float
    min_size: int
    rate_thresh: float


@dataclass(frozen=True)
class Gate:
    start: int
    end: int

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


class RunResult(NamedTuple):
    file_index: int
    result: GateResult | ErrorResult


def get_min_events(p: Params, sop: int) -> int:
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
    large_gap = tdiff_norm > p.gap_limit
    non_mono = tdiff < -p.non_mono_limit

    return np.select(
        [large_neg, large_gap, non_mono],
        [1, 2, 3],
        default=0,
    )


def get_anomaly_gates(anomalies: npt.NDArray[np.int64]) -> list[AnomalyGate]:
    total = anomalies.size
    if anomalies.sum() == 0:
        return [AnomalyGate(0, total - 1, None)]
    else:
        mask = anomalies > 1
        anomaly_positions = np.where(mask)[0].tolist()
        starts: list[int] = [0, *anomaly_positions]
        ends: list[int] = [*anomaly_positions, total - 1]
        codes: list[int | None] = [*anomalies[mask].tolist(), None]
        return [AnomalyGate(s, e, c) for s, e, c in zip(starts, ends, codes)]


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
    p: Params, i0: int, i1: int, tdiff: "pd.Series[float]"
) -> list[Gate]:
    # NOTE add 1 since the first diff value for this gate is not differentiable
    gates = find_flat(p, [], i0 + 1, i1, tdiff)
    if len(gates) == 0:
        return [Gate(i0, i1)]
    else:
        starts = [i0, *gates]
        ends = [*gates, i1]
        return [Gate(s, e) for s, e in zip(starts, ends)]


def get_all_gates(c: RunConfig) -> RunResult:
    # ASSUME all warnings are already triggered and captured elsewhere
    with warnings.catch_warnings(action="ignore"):
        fcs = fp.parse(c.path, channel_naming="$PnN")

    min_events = get_min_events(c.params, c.sop)
    t = fcs[1][c.channel]
    anomalies = get_anomalies(c.params, t)

    ano_gates = get_anomaly_gates(anomalies)
    valid_ano_gates = [g for g in ano_gates if g.length > min_events]

    if len(valid_ano_gates) == 0:
        res: GateResult | ErrorResult = ErrorResult(t)
    else:
        t_clean = t[anomalies == 0]
        tdiff_clean = t_clean - t_clean.shift(1)
        flat_gates = [
            fg
            for ag in valid_ano_gates
            for fg in get_flat_gates(c.params, ag.start, ag.end, tdiff_clean)
            if fg.length > min_events
        ]
        if len(flat_gates) == 0:
            res = ErrorResult(t)
        else:
            res = GateResult(
                valid_ano_gates, sorted(flat_gates, key=lambda g: g.length)
            )
    return RunResult(c.file_index, res)


def main(smk: Any) -> None:
    sp = smk.params
    p = Params(
        min_events=MinEvents(
            sop1=sp["min_event1"],
            sop2=sp["min_event2"],
            sop3=sp["min_event3"],
        ),
        spike_limit=sp["spike_limit"],
        gap_limit=sp["gap_limit"],
        non_mono_limit=sp["non_mono_limit"],
        rate_thresh=math.log10(100 / sp["rate_thresh"]),
        min_size=sp["min_size"],
    )
    channel_in = Path(smk.input["channels"])
    meta_in = Path(smk.input["clean_meta"])

    anomaly_out = Path(smk.output["anomaly"])
    flat_out = Path(smk.output["flat"])
    top_out = Path(smk.output["top"])
    event_out = Path(smk.output["events"])

    channels = read_time_channel_mapping(channel_in)

    META_COLUMNS = ["file_index", "org", "machine", "sop", "filepath"]

    df = pd.read_table(meta_in)[META_COLUMNS]
    runs = [
        RunConfig(p, file_index, Path(filepath), channels[(org, machine)], sop)
        for file_index, org, machine, sop, filepath in df.itertuples(index=False)
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
            fi = str(r.file_index)
            res = r.result

            def write_tsv_line(b: TextIO, xs: list[Any]) -> None:
                b.write("\t".join([fi, *[str(x) for x in xs]]) + "\n")

            if isinstance(res, GateResult):
                for ag in res.anomaly_gates:
                    write_tsv_line(ao, [ag.start, ag.end, ag.anomaly])
                for fg in res.flat_gates:
                    write_tsv_line(fo, [fg.start, fg.end])
                g0 = res.flat_gates[-1]
                write_tsv_line(to, [g0.start, g0.end])
            else:
                for ei, t in res.time.items():
                    write_tsv_line(eo, [ei, t])


main(snakemake)  # type: ignore

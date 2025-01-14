import pandas as pd
from pathlib import Path
from typing import NamedTuple, Any
from multiprocessing import Pool
from common.io import Delim, WritableFCS, ParsedEvents, with_fcs


class RunConfig(NamedTuple):
    file_index: int
    ipath: Path
    opath: Path
    start_time: float
    end_time: float


def gate_time_channel(c: RunConfig) -> None:
    def go(p: ParsedEvents) -> WritableFCS:
        df = p.events
        t = df["time"]
        mask = (c.start_time <= t) & (t < c.end_time)
        meta = p.meta.standard.serializable(set())
        return WritableFCS(meta, p.meta.params, {}, df[mask])

    with_fcs(c.ipath, c.opath, go, Delim(30), False, 12)


def main(smk: Any) -> None:
    gates_in = Path(smk.input["gates"])
    files_in = Path(smk.input["files"])

    files_out = Path(smk.output[0])

    out_dir = files_out.parent

    GATE_COLS = {
        "file_index": int,
        "event_start": int,
        "event_end": int,
        "time_start": float,
        "time_end": float,
        "valid": bool,
        "anomaly": int,
        "is_break": bool,
    }

    df_gates = pd.read_table(
        gates_in,
        names=list(GATE_COLS),
        dtype=GATE_COLS,
    ).set_index("file_index")
    df_gates = df_gates[df_gates["valid"]][["time_start", "time_end"]]

    df_files = pd.read_table(
        files_in,
        names=["file_index", "filepath"],
    ).set_index("file_index")

    df = df_gates.join(df_files)

    runs = [
        RunConfig(
            int(i),
            (fp := Path(filepath)),
            out_dir / fp.name,
            float(time_start),
            float(time_end),
        )
        for i, time_start, time_end, filepath in df.itertuples(index=True)
    ]

    with Pool(smk.threads) as p:
        p.map(gate_time_channel, runs)
        # list(map(standardize_fcs, [runs[0]]))

    with open(files_out, "wt") as f:
        for r in runs:
            f.write("\t".join([str(r.file_index), str(r.opath)]) + "\n")


main(snakemake)  # type: ignore

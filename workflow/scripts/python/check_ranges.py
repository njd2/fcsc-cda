import re
import warnings
import gzip
import pandas as pd
import fcsparser as fp  # type: ignore
from pathlib import Path
from typing import NamedTuple, TypeVar, Callable, Any
import datetime as dt
from itertools import groupby


class Result(NamedTuple):
    file_index: int
    channel: str
    n_over: int
    frac_over: float


def read_channel_ranges(meta: dict[str, str | int]) -> dict[str, int]:
    names = {
        int(m[1]): v
        for k, v in meta.items()
        if (m := re.match("\\$P([0-9]+)N", k)) is not None and isinstance(v, str)
    }
    return {
        names[int(m[1])]: int(v)
        for k, v in meta.items()
        if (m := re.match("\\$P([0-9]+)R", k)) is not None and isinstance(v, str)
    }


def check_ranges(file_index: int, path: Path, thresh: float) -> list[Result]:
    with warnings.catch_warnings(action="ignore"):
        meta, events = fp.parse(path, channel_naming="$PnN")

    n = len(events)
    rs = read_channel_ranges(meta)

    return [
        Result(
            file_index,
            name,
            (x := (events[name] > (rng * thresh)).sum()),
            x / n,
        )
        for name, rng in rs.items()
        if name != "time"
    ]


def main(smk: Any) -> None:
    thresh = smk.params["thresh"]
    clean_dir = Path(smk.params["clean_dir"])
    inpath = Path(smk.input[0])
    opath = Path(smk.output[0])

    df = pd.read_table(inpath)
    df = df[~df["group_has_any_error"]]

    results = [
        r
        for x in df[["file_index", "filepath"]].itertuples(index=False)
        for r in check_ranges(x[0], clean_dir / Path(x[1]).name, thresh)
    ]

    results.sort(key=lambda x: (x.file_index, x.channel))

    with gzip.open(opath, "wt") as f:
        for r in results:
            f.write(
                "\t".join(
                    [
                        str(int(r.file_index)),
                        r.channel,
                        str(r.n_over),
                        str(r.frac_over),
                    ]
                )
                + "\n"
            )


main(snakemake)  # type: ignore

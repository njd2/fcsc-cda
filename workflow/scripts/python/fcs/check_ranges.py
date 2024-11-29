import gzip
import pandas as pd
from pathlib import Path
from typing import NamedTuple, Any
from common.io import read_fcs, ParamKeyword, Ptype


class Result(NamedTuple):
    file_index: int
    channel: str
    n_over: int
    frac_over: float


def read_channel_ranges(ps: list[ParamKeyword]) -> dict[str, int]:
    names = {int(p.index_): p.value for p in ps if p.ptype is Ptype.NAME}
    return {names[p.index_]: int(p.value) for p in ps if p.ptype is Ptype.MAXRANGE}


def check_ranges(file_index: int, path: Path, thresh: float) -> list[Result]:
    p = read_fcs(path)

    n = len(p.events)
    rs = read_channel_ranges(p.meta.params)

    return [
        Result(
            file_index,
            name,
            (x := (p.events[name] > (rng * thresh)).sum()),
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

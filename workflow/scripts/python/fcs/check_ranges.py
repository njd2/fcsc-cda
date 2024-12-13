import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import NamedTuple, Any
from multiprocessing import Pool
from common.io import read_fcs, ParamKeyword, Ptype, write_fcs, Delim, WritableFCS


class Run(NamedTuple):
    file_index: int
    ipath: Path
    opath: Path
    thresh: float


class Result(NamedTuple):
    file_index: int
    channel: str
    n_over: int
    frac_over: float


def read_channel_ranges(ps: list[ParamKeyword]) -> dict[str, int]:
    names = {int(p.index_): p.value for p in ps if p.ptype is Ptype.NAME}
    return {names[p.index_]: int(p.value) for p in ps if p.ptype is Ptype.MAXRANGE}


def check_ranges(r: Run) -> list[Result]:
    p = read_fcs(r.ipath)

    n = len(p.events)
    rs = read_channel_ranges(p.meta.params)

    masks = {
        name: p.events[name] > (rng * r.thresh)
        for name, rng in rs.items()
        if name != "time"
    }

    all_mask = np.any(np.array([m.values for m in masks.values()]), axis=0)

    wfcs = WritableFCS(
        p.meta.standard.serializable(set()),
        p.meta.params,
        {},
        p.events[~all_mask],
    )

    write_fcs(r.opath, wfcs, Delim(30), False, 12)

    return [
        Result(r.file_index, name, (x := mask.sum()), x / n)
        for name, mask in masks.items()
    ]


def main(smk: Any) -> None:
    thresh = smk.params["thresh"]
    clean_dir = Path(smk.params["clean_dir"])
    inpath = Path(smk.input[0])
    files_opath = Path(smk.output["files"])
    results_opath = Path(smk.output["results"])

    df = pd.read_table(inpath)

    runs = [
        Run(
            int(x[0]),
            clean_dir / (n := Path(x[1]).name),
            files_opath.parent / n,
            thresh,
        )
        for x in df.itertuples(index=False)
    ]

    with Pool(smk.threads) as pl:
        results = pl.map(check_ranges, runs)

    flat_results = [y for xs in results for y in xs]
    flat_results.sort(key=lambda x: (x.file_index, x.channel))

    with gzip.open(results_opath, "wt") as f:
        for r in flat_results:
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

    with open(files_opath, "w") as f:
        for run in runs:
            f.write("\t".join(map(str, [run.file_index, run.opath])) + "\n")


main(snakemake)  # type: ignore

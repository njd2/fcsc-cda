import gzip
import math
import numpy.typing as npt
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from typing import NamedTuple, Any
from multiprocessing import Pool
from common.io import read_fcs


class RunConfig(NamedTuple):
    file_index: int
    path: Path
    embedding_sizes: list[int]
    delays: list[int]


class PEResult(NamedTuple):
    channel: str
    perm_entropy: float
    embedding_size: int
    delay: int


class RunResult(NamedTuple):
    file_index: int
    result: list[PEResult]


def perm_entropy(x: npt.NDArray[np.float32], d: int, tau: int) -> float:
    ntype = np.uint32
    n = x.shape[0]
    perms = [x[i : n - (d - i - 1) : tau] for i in range(d)]
    acc = np.array([0], dtype=ntype)
    for i, j in combinations(range(d), 2):
        acc = (acc << 1).astype(ntype) | (perms[i] <= perms[j]).astype(ntype)
    counts = np.bincount(acc)
    px = counts[np.nonzero(counts)[0]] / n
    perm_entropy: float = -np.nansum(px * np.log2(px))
    return perm_entropy / math.log2(math.factorial(d))


def compute_stuff(c: RunConfig) -> RunResult:
    parsed = read_fcs(c.path)
    df = parsed.events
    xs = [
        PEResult(col, perm_entropy(df[col]._values, d, tau), d, tau)
        for col in df.columns.tolist()
        for d in c.embedding_sizes
        for tau in c.delays
    ]
    return RunResult(c.file_index, xs)


def main(smk: Any) -> None:
    sp = smk.params
    embedding_sizes = sp["embedding_sizes"]
    delays = sp["delays"]
    files_in = Path(smk.input[0])

    entropy_out = Path(smk.output[0])

    df = pd.read_table(files_in)
    runs = [
        RunConfig(i, Path(p), embedding_sizes, delays)
        for i, p in df.itertuples(index=False)
    ]

    # weeeeeeeee
    with Pool(smk.threads) as pl:
        results = pl.map(compute_stuff, runs)

    with gzip.open(entropy_out, "wt") as o:
        for r in results:
            o.write("\t".join(map(str, [*r])) + "\n")


main(snakemake)  # type: ignore

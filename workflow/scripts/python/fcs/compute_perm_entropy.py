import gzip
import math
import numpy.typing as npt
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations, combinations
from typing import NamedTuple, Any
from multiprocessing import Pool
from common.io import read_fcs
from numba import njit  # type: ignore


class RunConfig(NamedTuple):
    file_index: int
    path: Path
    embedding_sizes: list[int]
    delays: list[int]


class PEResult(NamedTuple):
    channel: str
    weighted: bool
    perm_entropy: float
    embedding_size: int
    delay: int


class RunResult(NamedTuple):
    file_index: int
    result: list[PEResult]


# def perm_entropy(x: npt.NDArray[np.float32], d: int, tau: int) -> float:
#     ntype = np.uint32
#     n = x.shape[0]
#     perms = [x[i : n - (d - i - 1) : tau] for i in range(d)]
#     acc = np.array([0], dtype=ntype)
#     for i, j in combinations(range(d), 2):
#         acc = (acc << 1).astype(ntype) | (perms[i] <= perms[j]).astype(ntype)
#     counts = np.bincount(acc)
#     px = counts[np.nonzero(counts)[0]] / n
#     perm_entropy: float = -np.nansum(px * np.log2(px))
#     return perm_entropy / math.log2(math.factorial(d))


def perm_map(d: int) -> npt.NDArray[np.uint32]:
    perms = np.array(list(permutations(range(d)))).T
    acc = np.array([0], dtype=np.uint32)
    # TODO not dry
    for i, j in combinations(range(d), 2):
        acc = (acc << 1).astype(np.uint32) | (perms[i] <= perms[j]).astype(np.uint32)
    acc = np.sort(acc)
    mapping = np.zeros(max(acc) + 1, np.uint32)
    for i in range(acc.shape[0]):
        mapping[acc[i]] = i
    return mapping


# only support 2 <= d <= 8
PERM_INDICES: list[tuple[int, npt.NDArray[np.uint32]]] = [(0, np.array([0]))] * 2 + [
    (math.factorial(x), perm_map(x)) for x in range(2, 9)
]


@njit
def count_elements(imap, arr, counts):  # type: ignore
    for i in range(arr.shape[0]):
        counts[imap[arr[i]]] += 1
    return counts


@njit
def count_elements_weighted(imap, arr, counts, weights):  # type: ignore
    for i in range(arr.shape[0]):
        counts[imap[arr[i]]] += weights[i]
    return counts


def perm_entropy(
    x: npt.NDArray[np.float32],
    d: int,
    tau: int,
    weighted: bool,
) -> float:
    f, imap = PERM_INDICES[d]
    n = x.shape[0]
    perms = np.array([x[i : n - (d - i - 1) : tau] for i in range(d)])
    acc = np.array([0], dtype=np.uint32)
    for i in range(d - 1):
        for j in range(i + 1, d):
            acc = (acc << 1).astype(np.uint32) | (perms[i, :] <= perms[j, :]).astype(
                np.uint32
            )
    counts = np.zeros(f, dtype=np.uint64)
    if weighted:
        weights = perms.var(0)
        px = count_elements_weighted(imap, acc, counts, weights) / weights.sum()
    else:
        px = count_elements(imap, acc, counts) / n
    px[px == 0] = np.nan
    perm_entropy: float = -np.nansum(px * np.log2(px))
    return perm_entropy / math.log2(f)


def compute_stuff_inner(c: RunConfig, weighted: bool) -> list[PEResult]:
    parsed = read_fcs(c.path)
    df = parsed.events
    # time is special since we need to compute entropy on the differences
    t = df["time"]._values
    tdiff = t[1 : len(t) - 1] - t[0 : len(t) - 2]
    t_results = [
        PEResult("time", weighted, perm_entropy(tdiff, d, tau, weighted), d, tau)
        for d in c.embedding_sizes
        for tau in c.delays
    ]
    # and the rest are straightforward
    xs = [
        PEResult(col, weighted, perm_entropy(df[col]._values, d, tau, weighted), d, tau)
        for col in df.columns.tolist()
        if col != "time"
        for d in c.embedding_sizes
        for tau in c.delays
    ]
    return t_results + xs


def compute_stuff(c: RunConfig) -> RunResult:
    return RunResult(
        c.file_index,
        compute_stuff_inner(c, True) + compute_stuff_inner(c, True),
    )


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
            i = r.file_index
            es = r.result
            for e in es:
                o.write("\t".join(map(str, [i, *e])) + "\n")


main(snakemake)  # type: ignore

import gzip
import numpy.typing as npt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import NamedTuple, Any
from scipy.signal import lombscargle as ls  # type: ignore
from multiprocessing import Pool
from common.io import read_fcs

# For each channel, compute the power spectrum using the Lomb-Scargle method
# (which is necessary in this case since we don't have evenly spaced time-series
# data). Output spectra will be normalized such that power is dimensionless and
# between 0 and 1, which will allow direct comparison b/t channels/samples.

# skip height since it will likely have similar information compared to the
# area scatter channels; also not all files have it
COLUMNS = ["fsc_a", "ssc_a", "v450", "v500", "fitc", "pc55", "pe", "pc7", "apc", "ac7"]


class RunConfig(NamedTuple):
    file_index: int
    path: Path
    freqs: npt.NDArray[np.float32]


class RunResult(NamedTuple):
    file_index: int
    spectra: list[npt.NDArray[np.float64]]


def compute_stuff(c: RunConfig) -> RunResult | None:
    parsed = read_fcs(c.path)
    ts = parsed.meta.standard.timestep
    if ts is None:
        return None
    df = parsed.events
    t = (df["time"] * ts).values
    res = [
        ls(t, df[col].values, c.freqs, precenter=True, normalize=True)
        for col in COLUMNS
    ]
    return RunResult(c.file_index, res)


def main(smk: Any) -> None:
    sp = smk.params
    max_freq = float(sp["max_freq"])
    freq_steps = int(sp["freq_steps"])
    files_in = Path(smk.input[0])

    spectra_out = Path(smk.output[0])

    # incoming frequency parameters are in Hz, and lomb/scargle is written in
    # terms of rad/s
    freqs = np.linspace(0.0001, max_freq, freq_steps) * 2 * np.pi

    df = pd.read_table(files_in)
    runs = [RunConfig(i, Path(p), freqs) for i, p in df.itertuples(index=False)]

    # weeeeeeeee
    with Pool(smk.threads) as pl:
        results = pl.map(compute_stuff, runs)

    with gzip.open(spectra_out, "wt") as o:
        for r in results:
            if r is not None:
                fi = r.file_index
                # ASSUME each "result" is a list of vectors which correspond to
                # each channel, and that this list is the same length, always
                # the same order, and all vectors are 'freq_steps' long
                ss = r.spectra
                for i in range(freq_steps):
                    o.write(
                        "\t".join(map(str, [fi, freqs[i], *[s[i] for s in ss]])) + "\n"
                    )


main(snakemake)  # type: ignore

import re
import struct
import gzip
import math
import warnings
import numpy.typing as npt
import numpy as np
import pandas as pd
import fcsparser as fp  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any, TextIO
from dataclasses import dataclass
from multiprocessing import Pool

# Convert FCS files into "standard format" (as defined by this pipeline) which
# includes filtering to "valid" events and choosing only the channels necessary
# for downstream analysis.
#
# Doing this we make several assumptions and design choices:
# - There are no analysis segments (set these offsets to 0)
# - Ignore STEXT (set these offsets to zero)
# - Output will always be in float32 (is assume nothing is larger than 32bit)
# - Byte order will always be saved in little-endian (thanks Intel)
# - The only data stored in the new HEADER will be TEXT offsets and version;
#   this is to work around the fact that some FCS files are too big to have
#   their DATA offsets stored in the header (we store these in the TEXT segment
#   instead)

# record separator control code, basically guaranteed not to be in any of the
# keywords or values
DELIM = b"\x1e"

BEGINDATA_KEY = DELIM + b"DATASTART" + DELIM
ENDDATA_KEY = DELIM + b"DATAEND" + DELIM

# allow the two datastart/end fields to hold values up to 999,999,999,999 (that
# should be enough...hopefully)
DATAVALUE_LENGTH = 12

# length of the bytes in the BEGIN/ENDDATA fields
DATATEXT_LENGTH = len(BEGINDATA_KEY) + len(ENDDATA_KEY) + 1 + DATAVALUE_LENGTH * 2

# it seems most machines either use 58 bytes or 256 bytes. 58 is shorter ;)
HEADER_LENGTH = 6 + 4 + 8 * 6

# empty FCS header field padded to 8 bytes
EMPTY = " " * 7 + "0"

# TEXT keys that are required
REQUIRED_NON_P_KEYWORDS = [
    "$BEGINANALYSIS",
    "$BEGINDATA",
    "$BEGINSTEXT",
    "$BYTEORD",
    "$DATATYPE",
    "$ENDANALYSIS",
    "$ENDDATA",
    "$ENDSTEXT",
    "$MODE",
    "$NEXTDATA",
    "$PAR",
    "$TOT",
]


class RunConfig(NamedTuple):
    file_index: int
    ipath: Path
    opath: Path
    channel_map: dict[str, str]
    start: int
    end: int


ChannelMap = dict[str, str]
TextKWs = dict[str, Any]


def split_meta(meta: TextKWs) -> tuple[TextKWs, TextKWs]:
    nonrequired = {
        k: v
        for k, v in meta.items()
        if k != "__header__" and k not in REQUIRED_NON_P_KEYWORDS
    }
    params = {
        k: v for k, v in nonrequired.items() if re.match("\\$P[0-9]+", k) is not None
    }
    non_params = {k: v for k, v in nonrequired.items() if k not in params}
    return params, non_params


def clean_parameters(e: TextKWs, c: ChannelMap) -> TextKWs:
    split_params: list[tuple[int, str, Any]] = [
        (int(m[1]), m[2], v)
        for k, v in e.items()
        if (m := re.match("\\$P([0-9]+)(.)", k)) is not None
    ]
    index_map: dict[int, int] = {
        old: new + 1 for new, old in enumerate(i for i, _, v in split_params if v in c)
    }
    return {f"$P{index_map[i]}{t}": v for i, t, v in split_params if i in index_map}


def format_header(version: str, textlen: int) -> str:
    assert len(version) == 6, "version must be 6 bytes"
    # 6 bytes for version + 4 spaces + 8 byte fields for offsets (6 in total)
    return f"{version}    {HEADER_LENGTH:>8}{HEADER_LENGTH + textlen:>8}" + EMPTY * 4


def format_keywords(xs: TextKWs) -> bytes:
    return DELIM.join([k.encode() + DELIM + str(v).encode() for k, v in xs.items()])


def standardize_fcs(c: RunConfig) -> None:
    meta, df = fp.parse(c.ipath, channel_naming="$PnN")

    new_df = df[c.start : c.end, [*c.channel_map]]

    # this is in binary for some reason
    version = meta["__header__"]["FCS format"].decode()

    n_params = len(c.channel_map)
    binary_format = f"<{n_params}f"

    required: TextKWs = {
        "$BEGINANALYSIS": 0,
        "$ENDANALYSIS": 0,
        "$BEGINSTEXT": 0,
        "$ENDSTEXT": 0,
        "$BYTEORD": "1,2,3,4",
        "$DATATYPE": "F",
        "$MODE": "L",
        "$NEXTDATA": 0,
        "$PAR": n_params,
        "$TOT": len(new_df),
    }

    params, non_params = split_meta(meta)
    new_params = clean_parameters(params, c.channel_map)

    new_text = (
        format_keywords(required)
        + format_keywords(new_params)
        + format_keywords(non_params)
        + DELIM
    )

    text_length = DATATEXT_LENGTH + len(new_text)
    begindata = f"{HEADER_LENGTH + text_length:>DATAVALUE_LENGTH}".encode()
    enddata = f"{begindata + new_df.size * 4:>DATAVALUE_LENGTH}".encode()

    with open(c.opath, "wb") as f:
        # write new HEADER
        f.write(format_header(version, text_length).encode())
        # write new begin/end TEXT keywords
        f.write(BEGINDATA_KEY + begindata + ENDDATA_KEY + enddata + DELIM)
        # write the rest of the TEXT keywords
        f.write(new_text)
        # write all the data
        for r in new_df.itertuple(name=None):
            f.write(struct.pack(binary_format, *r))


def main(smk: Any) -> None:
    issues_in = Path(smk.input["issues"])
    top_gates_in = Path(smk.input["top"])


main(snakemake)  # type: ignore

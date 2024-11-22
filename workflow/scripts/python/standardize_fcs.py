import re
import struct
import warnings
import pandas as pd
import fcsparser as fp  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any
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
    # ASSUME all warnings are already triggered and captured elsewhere
    with warnings.catch_warnings(action="ignore"):
        meta, df = fp.parse(c.ipath, channel_naming="$PnN")

    new_df = df[c.start : c.end + 1][[*c.channel_map]]

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
    begindata_offset = HEADER_LENGTH + text_length
    begindata = f"{begindata_offset:0>{DATAVALUE_LENGTH}}".encode()
    enddata = f"{begindata_offset + new_df.size * 4:0>{DATAVALUE_LENGTH}}".encode()

    with open(c.opath, "wb") as f:
        # write new HEADER
        f.write(format_header(version, text_length - 1).encode())
        # write new begin/end TEXT keywords
        f.write(BEGINDATA_KEY + begindata + ENDDATA_KEY + enddata + DELIM)
        # write the rest of the TEXT keywords
        f.write(new_text)
        # write all the data
        for r in new_df.itertuples(name=None, index=False):
            f.write(struct.pack(binary_format, *r))


OrgMachChannelMap = dict[tuple[str, str], dict[str, str]]


# ASSUME all the channels are in a standardized order
def read_channel_mapping(p: Path) -> OrgMachChannelMap:
    acc: OrgMachChannelMap = {}
    with open(p, "r") as f:
        # skip header
        next(f, None)
        for i in f:
            s = i.rstrip().split("\t")
            org = s[0]
            machine = s[1]
            machine_name = s[2]
            std_name = s[3]
            key = (org, machine)
            if key not in acc:
                acc[key] = {}
            acc[key][machine_name] = std_name
    return acc


def main(smk: Any) -> None:
    top_gates_in = Path(smk.input["top"])
    channels_in = Path(smk.input["channels"])
    meta_in = Path(smk.input["meta"])

    flag_out = Path(smk.output[0])

    out_dir = flag_out.parent

    COLUMNS = ["org", "machine", "filepath", "start", "end"]

    ISSUE_COLUMNS = [
        "has_gain_variation",
        "has_voltage_variation",
        "missing_time",
        "missing_colors",
        "missing_scatter_area",
        "missing_scatter_height",
    ]

    df_meta = pd.read_table(meta_in).set_index("file_index")
    df_top_gates = pd.read_table(
        top_gates_in, names=["file_index", "start", "end"]
    ).set_index("file_index")

    df = df_top_gates.join(df_meta)
    df = df[~df[ISSUE_COLUMNS].any(axis=1)][COLUMNS]

    om_channel_map = read_channel_mapping(channels_in)

    runs = [
        RunConfig(
            (fp := Path(filepath)),
            out_dir / fp.name,
            om_channel_map[org, machine],
            start,
            end,
        )
        for org, machine, filepath, start, end in df.itertuples(index=False)
    ]

    with Pool(smk.threads) as p:
        p.map(standardize_fcs, runs)

    flag_out.touch()


main(snakemake)  # type: ignore

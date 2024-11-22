import re
import struct
import warnings
import pandas as pd
import fcsparser as fp  # type: ignore
from pathlib import Path
from typing import NamedTuple, Any, NewType
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

# export everything as 32-bit/little endian
PARAM_BITS = 32
BYTEORD = "1,2,3,4"
DATATYPE = "F"

# We are exporting data as an array so this has to be 'list'
MODE = "L"

# record separator control code, basically guaranteed not to be in any of the
# keywords or values
DELIM = b"\x1e"

# allow the two datastart/end fields to hold values up to 999,999,999,999 (that
# should be enough...hopefully)
DATAVALUE_LENGTH = 12

# it seems most machines either use 58 bytes or 256 bytes. 58 is shorter so why
# not? Note that the header consists of a version string (6 bytes) followed by 4
# spaces, followed by 6 ASCII numbers which are padded to 8 bytes long (total 58
# bytes)
HEADER_LENGTH = 58

_BEGINDATA_KEY = DELIM + b"$BEGINDATA" + DELIM
_ENDDATA_KEY = DELIM + b"$ENDDATA" + DELIM

# length of the bytes in the BEGIN/ENDDATA fields
_DATATEXT_LENGTH = len(_BEGINDATA_KEY) + len(_ENDDATA_KEY) + 1 + DATAVALUE_LENGTH * 2

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


TextValue = str | int
ParamIndex = NewType("ParamIndex", int)


class ExpandedParameter(NamedTuple):
    index_: ParamIndex
    name: str
    value: TextValue

    @property
    def key(self) -> str:
        return f"$P{self.index_}{self.name}"

    @property
    def to_serial(self) -> bytes:
        return self.key.encode() + DELIM + str(self.value).encode()


class ChannelNames(NamedTuple):
    param_index: ParamIndex
    short: str
    long: str

    def _to_param(self, name: str, value: TextValue) -> ExpandedParameter:
        return ExpandedParameter(self.param_index, name, value)

    @property
    def to_bits(self) -> ExpandedParameter:
        return self._to_param("B", PARAM_BITS)

    @property
    def to_short(self) -> ExpandedParameter:
        return self._to_param("N", self.short)

    @property
    def to_long(self) -> ExpandedParameter:
        return self._to_param("S", self.long)


ChannelMap = dict[str, ChannelNames]
OrgMachChannelMap = dict[tuple[str, str], ChannelMap]
TextKWs = dict[str, TextValue]


class RunConfig(NamedTuple):
    ipath: Path
    opath: Path
    channel_map: ChannelMap
    start: int
    end: int


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


def format_parameters(kws: TextKWs, cm: ChannelMap) -> bytes:
    # split parameters apart into a sane format that python can understand
    split_params = [
        ExpandedParameter(ParamIndex(int(m[1])), m[2], v)
        for k, v in kws.items()
        if (m := re.match("\\$P([0-9]+)(.)", k)) is not None
    ]
    # build a mapping between the current indices and the new indices/names; note
    # that that ChannelMap should have a mapping between the $PnN value and
    # the standardized index, $PnN, and $PnS values
    index_map: dict[ParamIndex, ChannelNames] = {
        p.index_: cm[p.value]
        for p in split_params
        if isinstance(p.value, str) and p.value in cm and p.name == "N"
    }
    # reindex all parameters except for $PnS, $PnN, and $PnB
    reindexed = [
        ExpandedParameter(index_map[p.index_].param_index, p.name, p.value)
        for p in split_params
        if p.index_ in index_map and p.name not in "NSB"
    ]
    # rebuild the $PnS and $PnN parameters (some of these might not be present
    # so easier to build from scratch and add rather than selectively
    # replace/add)
    short_names = [x.to_short for x in index_map.values()]
    long_names = [x.to_long for x in index_map.values()]
    # rebuild bits to guarantee standardized format
    bits = [x.to_bits for x in index_map.values()]
    # sort by index/type and serialize
    new_params = reindexed + short_names + long_names + bits
    new_params.sort(key=lambda x: (x.index_, x.name))
    return DELIM.join(x.to_serial for x in new_params)


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
        "$BYTEORD": BYTEORD,
        "$DATATYPE": DATATYPE,
        "$MODE": MODE,
        "$NEXTDATA": 0,
        "$PAR": n_params,
        "$TOT": len(new_df),
    }

    params, non_params = split_meta(meta)

    new_text = (
        format_keywords(required)
        + DELIM
        + format_parameters(params, c.channel_map)
        + DELIM
        + format_keywords(non_params)
        + DELIM
    )

    text_length = _DATATEXT_LENGTH + len(new_text)
    begindata_offset = HEADER_LENGTH + text_length
    begindata = f"{begindata_offset:0>{DATAVALUE_LENGTH}}".encode()
    enddata = f"{begindata_offset + new_df.size * 4:0>{DATAVALUE_LENGTH}}".encode()

    with open(c.opath, "wb") as f:
        # write new HEADER
        f.write(format_header(version, text_length - 1).encode())
        # write new begin/end TEXT keywords
        f.write(_BEGINDATA_KEY + begindata + _ENDDATA_KEY + enddata + DELIM)
        # write the rest of the TEXT keywords
        f.write(new_text)
        # write all the data
        for r in new_df.itertuples(name=None, index=False):
            f.write(struct.pack(binary_format, *r))


# ASSUME all the channels are in a standardized order
def read_channel_mapping(p: Path) -> OrgMachChannelMap:
    acc: OrgMachChannelMap = {}
    param_index = 1
    with open(p, "r") as f:
        # skip header
        next(f, None)
        for i in f:
            s = i.rstrip().split("\t")
            org = s[0]
            machine = s[1]
            machine_name = s[2]
            std_name = s[3]
            std_name_long = s[4]
            key = (org, machine)
            if key not in acc:
                param_index = 1
                acc[key] = {}
            new = ChannelNames(ParamIndex(param_index), std_name, std_name_long)
            acc[key][machine_name] = new
            param_index = param_index + 1
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
        # list(map(standardize_fcs, [runs[0]]))

    flag_out.touch()


main(snakemake)  # type: ignore

import re
import struct
import warnings
from pathlib import Path
from typing import NamedTuple, Callable, NewType, Any, TypeVar, assert_never
import pandas as pd
import fcsparser as fp  # type: ignore
from pydantic import BaseModel as BaseModel_, validator
from dataclasses import dataclass
from enum import Enum

X = TypeVar("X")
Y = TypeVar("Y")

# When writing FCS files we make several assumptions/choices
# - There are no analysis segments (set these offsets to 0)
# - Ignore STEXT (set these offsets to zero)
# - Output will always be in float32 (is assume nothing is larger than 32bit)
# - Byte order will always be saved in little-endian (thanks Intel)
# - The only data stored in the new HEADER will be TEXT offsets and version;
#   this is to work around the fact that some FCS files are too big to have
#   their DATA offsets stored in the header (we store these in the TEXT segment
#   instead)
# - version will be fixed to 3.1

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

# version (assumped to be 6 chars)
VERSION = "FCS3.1"

TextValue = str | int
TextKWs = dict[str, TextValue]
ParamIndex = NewType("ParamIndex", int)


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


class BaseModel(BaseModel_):
    class Config:
        validate_default = True
        extra = "forbid"
        frozen = True


class Version(Enum):
    v3_1 = "3.1"
    v3_0 = "3.0"

    def choose(self, a: X, b: X) -> X:
        if self is Version.v3_0:
            return a
        elif self is Version.v3_1:
            return b


# NOTE technically 3.0 allows mixed, but hopefully we never find these
class Byteord(Enum):
    LITTLE = "1,2,3,4"
    BIG = "4,3,2,1"


class Datatype(Enum):
    ASCII = "A"
    INT = "I"
    FLOAT = "F"
    DOUBLE = "D"


class Mode(Enum):
    LIST = "L"
    # NOTE technically these aren't allowed in 3.1
    UNCOR = "U"
    COR = "C"


class Originality(Enum):
    ORIGINAL = "Original"
    NONDATAMODIFIED = "NonDataModified"
    APPENDED = "Appended"
    DATAMODIFIED = "DataModified"


class TEXTRequired(BaseModel_):
    beginanalysis: int
    beginstext: int
    begindata: int
    endanalysis: int
    endstext: int
    enddata: int
    byteord: Byteord
    datatype: Datatype
    mode: Mode
    nextdata: int
    par: int
    tot: int

    @property
    def line(self) -> list[str]:
        return [
            x.value if isinstance(x, Enum) else str(x) for x in self.dict().values()
        ]


class TEXTCommon(TEXTRequired):
    abrt: int | None
    btim: str | None  # hh:mm:ss[.cc]
    cells: str | None
    com: str | None
    # NOTE CSMODE, CSVBITS, and CSVnFLAG are analysis fields that we probably
    # don't want
    cyt: str | None
    cytsn: str | None
    date: str | None  # formatted like dd-mmm-yyyy (although many don't do this)
    etim: str | None  # hh:mm:ss[.cc]
    exp: str | None
    fil: str | None
    gate: int | None
    gating: str | None
    inst: str | None
    lost: int | None
    op: str | None
    proj: str | None
    smno: str | None
    src: str | None
    sys: str | None
    timestep: float
    vol: float | None
    tr: str | None

    @property
    def line(self) -> list[str]:
        return super().line + [
            "" if x is None else str(x) for x in self.dict().values()
        ]


class _TEXT3_1(BaseModel):
    spillover: str | None
    originality: Originality | None
    last_modified: str | None
    last_modifier: str | None
    plateid: str | None
    platename: str | None
    wellid: str | None


class TEXT3_1(TEXTCommon, _TEXT3_1):

    @property
    def line(self) -> list[str]:
        return (
            super().line
            + [
                "" if x is None else (x.value if isinstance(x, Enum) else str(x))
                for x in self.dict().values()
            ]
            + [""] * 2
        )


class _TEXT3_0(BaseModel):
    comp: str | None
    unicode: str | None


class TEXT3_0(TEXTCommon, _TEXT3_0):

    @property
    def line(self) -> list[str]:
        return (
            super().line
            + [""] * 7
            + ["" if x is None else str(x) for x in self.dict().values()]
        )


TEXT_HEADER = (
    [*TEXTCommon.__fields__.keys()]
    + [*_TEXT3_1.__fields__.keys()]
    + [*_TEXT3_0.__fields__.keys()]
)


PARAM_TYPES_3_0 = [
    "B",
    "E",
    "R",
    "N",
    "F",
    "G",
    "L",
    "O",
    "P",
    "S",
    "T",
    "V",
]

PARAM_TYPES_3_1 = [*PARAM_TYPES_3_0, "CALIBRATION", "D"]


def build_param_re(xs: list[str]) -> re.Pattern[str]:
    # FSC keywords are case insensitive
    ys = [x.lower() for x in xs]
    return re.compile(f"\\$p([0-9]+)({"|".join(ys)})")


PARAM_RE_3_0 = build_param_re(PARAM_TYPES_3_0)
PARAM_RE_3_1 = build_param_re(PARAM_TYPES_3_1)


class AmpType(NamedTuple):
    decades: float
    zero: float


class Calibration(NamedTuple):
    value: float
    unit: str


class LinDisplay(NamedTuple):
    lower: float
    upper: float


class LogDisplay(NamedTuple):
    decades: float
    offset: float


Display = LinDisplay | LogDisplay


class ParsedParam(BaseModel):
    b: int
    e: AmpType
    r: int
    n: str
    f: str | None
    g: float | None
    l: list[int]
    o: int | None
    p: int | None
    s: str | None
    t: str | None
    v: float | None
    # 3.1 only
    calibration: Calibration | None
    d: Display | None

    @validator("e", pre=True)
    def validate_amp_type(cls, v: Any) -> AmpType:
        assert isinstance(v, str), "amp type must be string"
        m = re.match("([0-9.]+),([0-9.]+)", v)
        assert m is not None, "amp type must be like 'f1,f2'"
        return AmpType(float(m[1]), float(m[2]))

    @validator("calibration", pre=True)
    def validate_calibration(cls, v: Any) -> Calibration | None:
        if v is None:
            return None
        assert isinstance(v, str), "calibration must be string"
        m = re.match("([0-9.]+),(.+)", v)
        assert m is not None, "calibration must be like 'f1,string'"
        return Calibration(float(m[1]), m[2])

    @validator("d", pre=True)
    def validate_display(cls, v: Any) -> Display | None:
        if v is None:
            return None
        assert isinstance(v, str), "display must be string"
        m = re.match("(.+),([0-9.]+),([0-9.]+)", v)
        assert m is not None, "display must be like 'string,f1,f2'"
        if m[1] == "Linear":
            return LinDisplay(float(m[2]), float(m[3]))
        elif m[2] == "Logarithmic":
            return LogDisplay(float(m[2]), float(m[3]))
        else:
            assert False, "must be either linear or logarithmic"

    @classmethod
    def header(cls) -> list[str]:
        return [
            "bits",
            "log_decades",
            "log_zero",
            "maxrange",
            "shortname",
            "filtername",
            "gain",
            "wavelength",
            "power",
            "percent_emitted",
            "detector_type",
            "detector_voltage",
            "calibration_value",
            "calibration_type",
            "display_type",
            "display_v1",
            "display_v2",
        ]

    @property
    def amp_type_line(self) -> list[str]:
        e = self.e
        return [str(e.decades), str(e.zero)] if e is not None else ["", ""]

    @property
    def calibration_line(self) -> list[str]:
        c = self.calibration
        return [str(c.value), c.unit] if c is not None else ["", ""]

    @property
    def display_line(self) -> list[str]:
        d = self.d
        if isinstance(d, LinDisplay):
            return ["lin", str(d.lower), str(d.upper)]
        elif isinstance(d, LogDisplay):
            return ["log", str(d.decades), str(d.offset)]
        elif d is None:
            return [""] * 3
        else:
            assert_never(d)

    @property
    def line(self) -> list[str]:
        xs = [
            self.b,
            *self.amp_type_line,
            self.r,
            self.n,
            self.f,
            self.g,
            self.l,
            self.o,
            self.p,
            self.t,
            self.v,
            *self.calibration_line,
            *self.display_line,
        ]
        return ["" if x is None else str(x) for x in xs]


class ParamKeyword(NamedTuple):
    """A keyword describing a parameter (eg "$PnN")"""

    index_: ParamIndex
    ptype: str
    value: TextValue

    @property
    def key(self) -> str:
        return f"$P{self.index_}{self.ptype}"

    @property
    def to_serial(self) -> bytes:
        return self.key.encode() + DELIM + str(self.value).encode()


class FCSHeader(NamedTuple):
    text_start: int
    text_end: int
    analysis_start: int
    analysis_end: int
    data_start: int
    data_end: int


@dataclass(frozen=True)
class ParsedTEXT:
    header: FCSHeader
    standard: TEXT3_0 | TEXT3_1
    params: list[ParamKeyword]
    nonstandard: dict[str, str]


AnyTEXT = TEXT3_0 | TEXT3_1


@dataclass(frozen=True)
class FCSMetadata:
    meta: ParsedTEXT
    warnings: list[str]


@dataclass(frozen=True)
class FCSEvents(FCSMetadata):
    events: pd.DataFrame


class FCSWritable(NamedTuple):
    params: list[ParamKeyword]
    other: TextKWs
    events: pd.DataFrame


def format_header(textlen: int) -> str:
    # 6 bytes for version + 4 spaces + 8 byte fields for offsets (6 in total)
    return f"{VERSION}    {HEADER_LENGTH:>8}{HEADER_LENGTH + textlen:>8}" + EMPTY * 4


def format_parameters(ps: list[ParamKeyword]) -> bytes:
    return DELIM.join(x.to_serial for x in ps)


def format_keywords(xs: TextKWs) -> bytes:
    return DELIM.join([k.encode() + DELIM + str(v).encode() for k, v in xs.items()])


def read_header(meta: dict[str, Any]) -> tuple[FCSHeader, Version]:
    h = meta["__header__"]
    r = FCSHeader(
        h["text start"],
        h["text end"],
        h["analysis start"],
        h["analysis end"],
        h["data start"],
        h["data end"],
    )
    return (r, Version(h["FCS format"].decode()[3:]))


def split_meta(meta: dict[str, Any]) -> ParsedTEXT:
    header, version = read_header(meta)
    # convert all to lowercase (keywords are not case sensitive); also remove
    # gating parameters (for now) since idk what to do with them
    text = {
        k.lower(): v
        for k, v in meta.items()
        if k != "__header__" or re.match("\\$(G|R|Pk|PKN)[0-9]+.", k) is None
    }
    print(text["$flowrate"])
    pat = version.choose(PARAM_RE_3_0, PARAM_RE_3_1)
    params = {
        k: ParamKeyword(ParamIndex(int(m[1])), m[2], v)
        for k, v in text.items()
        if (m := re.match(pat, k)) is not None
    }
    nonparams = {k: v for k, v in text.items() if k not in params}
    standard = {
        m[1]: v
        for k, v in nonparams.items()
        if (m := re.match("\\$(.+)", k)) is not None
    }
    _text = version.choose(TEXT3_0, TEXT3_1)(**standard)
    nonstandard = {k: v for k, v in nonparams.items() if k not in standard}
    return ParsedTEXT(header, _text, [*params.values()], nonstandard)


def write_fcs(path: Path, p: FCSWritable) -> None:
    nrow, ncol = p.events.shape
    binary_format = f"<{ncol}f"

    required: TextKWs = {
        "$BEGINANALYSIS": 0,
        "$ENDANALYSIS": 0,
        "$BEGINSTEXT": 0,
        "$ENDSTEXT": 0,
        "$BYTEORD": BYTEORD,
        "$DATATYPE": DATATYPE,
        "$MODE": MODE,
        "$NEXTDATA": 0,
        "$PAR": ncol,
        "$TOT": nrow,
    }

    new_text = (
        format_keywords(required)
        + DELIM
        + format_parameters(p.params)
        + DELIM
        + format_keywords(p.other)
        + DELIM
    )

    text_length = _DATATEXT_LENGTH + len(new_text)
    begindata_offset = HEADER_LENGTH + text_length
    begindata = f"{begindata_offset:0>{DATAVALUE_LENGTH}}".encode()
    enddata = f"{begindata_offset + p.events.size * 4:0>{DATAVALUE_LENGTH}}".encode()

    with open(path, "wb") as f:
        # write new HEADER
        f.write(format_header(text_length - 1).encode())
        # write new begin/end TEXT keywords
        f.write(_BEGINDATA_KEY + begindata + _ENDDATA_KEY + enddata + DELIM)
        # write the rest of the TEXT keywords
        f.write(new_text)
        # write all the data
        for r in p.events.itertuples(name=None, index=False):
            f.write(struct.pack(binary_format, *r))


def read_fcs_metadata(p: Path) -> FCSMetadata:
    with warnings.catch_warnings(record=True, action="always") as w:
        meta = fp.parse(p, meta_data_only=True)
        warn_msgs = [str(x.message).replace("\n", " ") for x in w]
    _meta = split_meta(meta)
    return FCSMetadata(_meta, warn_msgs)


def read_fcs(p: Path) -> FCSEvents:
    with warnings.catch_warnings(record=True, action="always") as w:
        meta, data = fp.parse(p, channel_naming="$PnN")
        warn_msgs = [str(x.message).replace("\n", " ") for x in w]
    _meta = split_meta(meta)
    return FCSEvents(_meta, warn_msgs, data)


def with_fcs(ip: Path, op: Path, fun: Callable[[FCSEvents], FCSWritable]) -> None:
    p = read_fcs(ip)
    write_fcs(op, fun(p))

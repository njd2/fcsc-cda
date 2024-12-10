import re
import struct
import calendar
import warnings
import datetime as dt
from pathlib import Path
from typing import NamedTuple, Callable, NewType, Any, TypeVar, assert_never, Type, Self
import pandas as pd
import fcsparser as fp  # type: ignore
from pydantic import BaseModel as BaseModel_, validator
from pydantic import NonNegativeInt, NonNegativeFloat
from dataclasses import dataclass
from enum import StrEnum
from common.functional import fmap_maybe, from_maybe

X = TypeVar("X")
Y = TypeVar("Y")

# it seems most machines either use 58 bytes or 256 bytes. 58 is shorter so why
# not? Note that the header consists of a version string (6 bytes) followed by 4
# spaces, followed by 6 ASCII numbers which are padded to 8 bytes long (total 58
# bytes)
HEADER_LENGTH = 58

# Fields for begin/end data offsets, which need to be specially calculated and
# thus the field widths must be known in advance.
_BEGINDATA_KEY = b"$BEGINDATA"
_ENDDATA_KEY = b"$ENDDATA"
_BEGINDATA_KEY_LENGTH = len(b"$BEGINDATA")
_ENDDATA_KEY_LENGTH = len(b"$ENDDATA")

# empty FCS header field padded to 8 bytes
EMPTY = " " * 7 + "0"

# The index of a parameter (ie 'n' in '$PnN')
ParamIndex = NewType("ParamIndex", int)

# The keyword delimiter
Delim = NewType("Delim", int)

# A key with a dollar sign prefixed, which according to the spec means
# "standardized key"
StandardKey = NewType("StandardKey", str)


def make_standard_key(s: str) -> StandardKey:
    return StandardKey(f"${s}".upper())


class Version(StrEnum):
    """FCS version string"""

    v3_1 = "3.1"
    v3_0 = "3.0"

    def choose(self, a: X, b: X) -> X:
        if self is Version.v3_0:
            return a
        elif self is Version.v3_1:
            return b


class BaseModel(BaseModel_):
    class Config:
        validate_default = True
        extra = "forbid"
        frozen = True


#
# HEADER segment
#


class FCSHeader(NamedTuple):
    """
    The offsets in the FCS header.

    Note, the version is encoded elsewhere.
    """

    text_start: int
    text_end: int
    analysis_start: int
    analysis_end: int
    data_start: int
    data_end: int

    @property
    def line(self) -> list[str]:
        return [str(x) for x in self._asdict().values()]


#
# TEXT segment (fun fun)
#
# The basic idea is to use pydantic to encode the expected types we wish to
# parse, since in many cases we know the format that each type is supposed to
# have according to FCS3.0 and FCS3.1. Not all machines actually follow these,
# so some "bad behavior" is accommodated. By convention, each field in the
# pydnatic class will have a __str__ method that matches how it should be
# literally printed both in an FCS file and when dumped to a text-based output.
#
# Parameter keywords are dealt with separately since these are special.
#


class Byteord(StrEnum):
    """Allowed values for the $BYTEORD keyword"""

    # NOTE technically 3.0 allows mixed, but hopefully we never find these
    LITTLE = "1,2,3,4"
    BIG = "4,3,2,1"


class Datatype(StrEnum):
    """Allowed values for the $DATATYPE keyword"""

    # NOTE ASCII is not supported (because it's terrible and would require some
    # coding gymnastics to make work)
    INT = "I"
    FLOAT = "F"
    DOUBLE = "D"


class Mode(StrEnum):
    """Allowed values for the $MODE keyword"""

    LIST = "L"
    # NOTE technically these aren't allowed in 3.1
    UNCOR = "U"
    COR = "C"


class Originality(StrEnum):
    """Allowed values for the $ORIGINALITY keyword (3.1 only)"""

    ORIGINAL = "Original"
    NONDATAMODIFIED = "NonDataModified"
    APPENDED = "Appended"
    DATAMODIFIED = "DataModified"


def _print_time_centiseconds(t: dt.time, f: float) -> str:
    s = t.strftime("%H:%M:%S")
    c = int(t.microsecond * f)
    return f"{s}.{c:0>2}"


class FCSTime30(dt.time):
    """Time that is printed according to FCS 3.0 format"""

    def __str__(self) -> str:
        return _print_time_centiseconds(self, 60 / 1000000)


class FCSTime31(dt.time):
    """Time that is printed according to FCS 3.1 format"""

    def __str__(self) -> str:
        return _print_time_centiseconds(self, 1 / 10000)


class FCSDate(dt.date):
    """Date that is printed according to FCS 3.0/3.1 format"""

    def __str__(self) -> str:
        return self.strftime("%d-%b-%Y").upper()


class FCSDatetime(dt.datetime):
    """Datetime that is printed according to FCS 3.0/3.1 format"""

    def __str__(self) -> str:
        s = self.strftime("%d-%b-%Y %H:%M:%S").upper()
        c = int(self.microsecond / 10000)
        return f"{s}.{c:0>2}"


def month_to_int(s: str) -> int | None:
    _s = s.lower()
    # NOTE: month_abbr has a blank string at 0 so that the actual months start
    # at index 1
    return next(
        (i for i, m in enumerate(calendar.month_abbr) if m.lower() == _s),
        None,
    )


def make_date(year: str, month: str, day: str) -> FCSDate | None:
    return fmap_maybe(lambda m: FCSDate(int(year), m, int(day)), month_to_int(month))


def make_datetime(
    year: str,
    month: str,
    day: str,
    hour: str,
    minute: str,
    seconds: str,
    centiseconds: str | None,
) -> FCSDatetime | None:
    c = from_maybe(0, fmap_maybe(int, centiseconds))
    return fmap_maybe(
        lambda m: FCSDatetime(
            int(year),
            m,
            int(day),
            int(hour),
            int(minute),
            int(seconds),
            c * 10000,
        ),
        month_to_int(month),
    )


FCSTimeType = TypeVar("FCSTimeType", FCSTime30, FCSTime31)


def make_time(
    t: Type[FCSTimeType],
    hour: str,
    minute: str,
    seconds: str,
    centiseconds: str | None,
) -> FCSTimeType:
    c = from_maybe(0, fmap_maybe(int, centiseconds))
    return t(int(hour), int(minute), int(seconds), c * 10000)


def _validate_time(t: Type[FCSTimeType], v: Any) -> FCSTimeType | None:
    if v is None:
        return None
    if isinstance(v, t):
        return v
    if isinstance(v, str):
        r = fmap_maybe(
            lambda m: make_time(t, m[1], m[2], m[3], m[5]),
            re.match(TIME_RE, v),
        )
        if r is not None:
            return r
    assert False, f"time must be like HH:MM:SS[.CS], got {v}"


TIME_RE = re.compile("(\\d{2}):(\\d{2}):(\\d{2})(\\.(\\d{2}))?")

DATE_RE_YYYY_MMM_DD = re.compile("([0-9]{4})-([A-Za-z]{3})-([0-9]{2})")
DATE_RE_DD_MMM_YYYY = re.compile("([0-9]{2})-([A-Za-z]{3})-([0-9]{4})")

DATETIME_RE = re.compile(
    "(\\d{2})-([A-Za-z]{3})-(\\d{4}) (\\d{2}):(\\d{2}):(\\d{2})(\\.(\\d{2}))?"
)


class _IsTabulatable:
    """Superclass to represent FCS keywords that can be written to a table."""

    @classmethod
    def header(self) -> list[str]:
        """Return row header"""
        return NotImplemented

    @property
    def rowvalues(self) -> list[str]:
        """Return row values. Assumed to match exactly with header."""
        return NotImplemented

    @property
    def mapping(self) -> dict[str, str]:
        return {h: v for h, v in zip(self.header(), self.rowvalues)}


class _IsSerializable:
    """
    Superclass to represent FCS keywords that can be written to a new TEXT segment.
    """

    @property
    def keyval_pairs(self) -> dict[str, Any]:
        return NotImplemented

    @property
    def keywords(self) -> dict[StandardKey, str]:
        """Return keyword mapping with string values as they shall appear in the
        TEXT segment of the FCS file.
        """
        return {make_standard_key(k): str(v) for k, v in self.keyval_pairs.items()}

    def serialize(self, delim: Delim) -> bytes:
        """Return keywords serialized to bytes with desired delimiter."""
        d = delim.to_bytes(1)
        return d.join(k.encode() + d + v.encode() for k, v in self.keywords.items())


class TEXTMeta(BaseModel, _IsSerializable):
    """
    Keywords in the TEXT segment representing the data itself.

    This is separate to make it easier to represent writable FCS objects
    which either don't require these a priori.
    """

    beginanalysis: NonNegativeInt
    beginstext: NonNegativeInt
    endanalysis: NonNegativeInt
    endstext: NonNegativeInt
    byteord: Byteord
    datatype: Datatype
    mode: Mode
    nextdata: NonNegativeInt
    par: NonNegativeInt
    tot: NonNegativeInt

    @property
    def keyval_pairs(self) -> dict[str, str]:
        m = self.dict()
        return {f: str(x) for f in TEXTMeta.__fields__ if (x := m[f]) is not None}


class _TEXTDataOffsets(BaseModel):
    """Data start and end in TEXT segment.

    This is separate from everything else because these values are only ever
    read and cannot be specified a priori when serializing since we won't know
    what their values should be until after we make the entire TEXT segment.

    """

    begindata: NonNegativeInt
    enddata: NonNegativeInt


class Trigger(NamedTuple):
    name: str
    channel: int

    def __str__(self) -> str:
        return f"{self.name},{self.channel}"


class _TEXTCommon(BaseModel):
    """
    Keyword common between 3.0 and 3.1 standards.
    """

    abrt: int | None
    cells: str | None
    com: str | None
    # NOTE CSMODE, CSVBITS, and CSVnFLAG are analysis fields that we probably
    # don't want
    cyt: str | None
    cytsn: str | None
    date: FCSDate | None
    exp: str | None
    fil: str | None
    gate: NonNegativeInt | None
    gating: str | None
    inst: str | None
    lost: NonNegativeInt | None
    op: str | None
    proj: str | None
    smno: str | None
    src: str | None
    sys: str | None
    timestep: NonNegativeFloat | None
    vol: NonNegativeFloat | None
    tr: Trigger | None

    @validator("date", pre=True)
    def validate_date(cls, v: Any) -> FCSDate | None:
        # the date field is *supposed* to be dd-mmm-yyyy according to the spec,
        # but unfortunately many machines don't obey this convention :/
        if v is None:
            return None
        if isinstance(v, FCSDate):
            return v
        if isinstance(v, str):
            m1 = re.match(DATE_RE_YYYY_MMM_DD, v)
            if m1 is not None:
                return make_date(m1[1], m1[2], m1[3])
            m2 = re.match(DATE_RE_DD_MMM_YYYY, v)
            if m2 is not None:
                return make_date(m2[3], m2[2], m2[1])
        assert False, f"date must be YYYY-MMM-DD or DD-MMM-YYYY, got {v}"

    @validator("tr", pre=True)
    def validate_trigger(cls, v: Any) -> Trigger | None:
        if v is None:
            return None
        if isinstance(v, Trigger):
            return v
        if isinstance(v, str):
            m = re.match("(.+),([0-9]+)", v)
            if m is not None:
                return Trigger(m[1], int(m[2]))
        assert False, f"trigger must be like 'string,int', got {v}"

    @classmethod
    def _header_common(self) -> list[str]:
        return [x for x in _TEXTCommon.__fields__ if x not in ["date", "tr"]] + [
            "date",
            "trigger_name",
            "trigger_channel",
        ]

    @property
    def _values_common(self) -> list[str]:
        m = self.dict()
        xs = [
            "" if (x := m[f]) is None else str(x)
            for f in _TEXTCommon.__fields__
            if f not in ["tr", "date"]
        ]
        return [
            *xs,
            from_maybe("", fmap_maybe(dt.date.isoformat, self.date)),
            from_maybe("", fmap_maybe(lambda x: x.name, self.tr)),
            from_maybe("", fmap_maybe(lambda x: str(x.channel), self.tr)),
        ]


class _TEXT31(BaseModel):
    """
    3.1-specific keywords
    """

    btim: FCSTime31 | None
    etim: FCSTime31 | None
    # TODO make a standardized interface for this and $COMP
    spillover: str | None
    originality: Originality | None
    last_modified: FCSDatetime | None
    last_modifier: str | None
    plateid: str | None
    platename: str | None
    wellid: str | None

    @validator("last_modified", pre=True)
    def validate_datetime(cls, v: Any) -> FCSDatetime | None:
        if v is None:
            return None
        if isinstance(v, FCSDatetime):
            return v
        if isinstance(v, str):
            m = re.match(DATETIME_RE, v)
            if m is not None:
                return make_datetime(m[3], m[2], m[1], m[4], m[5], m[6], m[8])
        assert False, f"datetime must be like YYYY-MMM-DD HH:MM:SS(.CS), got {v}"

    @validator("etim", "btim", pre=True)
    def validate_time(cls, v: Any) -> FCSTime31 | None:
        return _validate_time(FCSTime31, v)

    @classmethod
    def _header_31(self) -> list[str]:
        return [*_TEXT31.__fields__]

    @property
    def _values_31(self) -> list[str]:
        m = self.dict()
        return [
            (
                ""
                if (x := m[f]) is None
                else (
                    dt.time.strftime(x, "%H:%M:%S.%f")
                    if isinstance(x, dt.time)
                    else (
                        dt.datetime.isoformat(x)
                        if isinstance(x, dt.datetime)
                        else str(x)
                    )
                )
            )
            for f in _TEXT31.__fields__
        ]


class _TEXT30(BaseModel):
    """
    3.0-specific keywords
    """

    btim: FCSTime30 | None
    etim: FCSTime30 | None
    comp: str | None
    unicode: str | None

    @validator("etim", "btim", pre=True)
    def validate_time(cls, v: Any) -> FCSTime30 | None:
        return _validate_time(FCSTime30, v)

    @classmethod
    def _header_30(self) -> list[str]:
        return [*_TEXT30.__fields__]

    @property
    def _values_30(self) -> list[str]:
        m = self.dict()
        return [
            (
                ""
                if (x := m[f]) is None
                else (
                    dt.time.strftime(x, "%H:%M:%S.%f")
                    if isinstance(x, dt.time)
                    else str(x)
                )
            )
            for f in _TEXT30.__fields__
        ]


class SerializableTEXT31(_TEXTCommon, _TEXT31, _IsSerializable):
    """
    Serializable non-param keywords specific to FCS 3.1 without the offset keywords.
    """

    @property
    def keyval_pairs(self) -> dict[str, str]:
        m = self.dict()
        return {
            k: v
            for k in [*_TEXTCommon.__fields__, *_TEXT31.__fields__]
            if (v := m[k]) is not None
        }


class SerializableTEXT30(_TEXTCommon, _TEXT30, _IsSerializable):
    """
    Serializable non-param keywords specific to FCS 3.0 without the offset keywords.
    """

    # TODO add method to convert 3.0 to 3.1

    @property
    def keyval_pairs(self) -> dict[str, str]:
        m = self.dict()
        return {
            k: v
            for k in [*_TEXTCommon.__fields__, *_TEXT30.__fields__]
            if (v := m[k]) is not None
        }


SerializableTEXT = SerializableTEXT30 | SerializableTEXT31


class _TabularTEXTCommon(_TEXTDataOffsets, TEXTMeta, _TEXTCommon):
    @classmethod
    def _header_tabcommon(self) -> list[str]:
        return [
            *_TEXTDataOffsets.__fields__,
            *TEXTMeta.__fields__,
            *self._header_common(),
        ]

    @property
    def _values_tabcommon(self) -> list[str]:
        m = self.dict()
        xs = [
            "" if (x := m[f]) is None else str(x)
            for f in [*_TEXTDataOffsets.__fields__, *TEXTMeta.__fields__]
        ]
        return [*xs, *self._values_common]


class TabularTEXT31(_TabularTEXTCommon, _TEXT31, _IsTabulatable):
    """
    Tabulatable non-param keywords specific to FCS 3.1.
    """

    @classmethod
    def header(self) -> list[str]:
        return self._header_tabcommon() + self._header_31()

    @property
    def rowvalues(self) -> list[str]:
        return self._values_tabcommon + self._values_31

    def serializable(self, exclude: set[str]) -> SerializableTEXT31:
        m = self.dict()
        return SerializableTEXT31(
            **{f: m[f] for f in SerializableTEXT31.__fields__ if f not in exclude}
        )


class TabularTEXT30(_TabularTEXTCommon, _TEXT30, _IsTabulatable):
    """
    Tabulatable non-param keywords specific to FCS 3.0.
    """

    @classmethod
    def header(self) -> list[str]:
        return self._header_tabcommon() + self._header_30()

    @property
    def rowvalues(self) -> list[str]:
        return self._values_tabcommon + self._values_30

    def serializable(self, exclude: set[str]) -> SerializableTEXT30:
        m = self.dict()
        return SerializableTEXT30(
            **{f: m[f] for f in SerializableTEXT30.__fields__ if f not in exclude}
        )


TabularTEXT = TabularTEXT30 | TabularTEXT31

StandardKeys30 = [make_standard_key(x) for x in TabularTEXT30.__fields__]
StandardKeys31 = [make_standard_key(x) for x in TabularTEXT31.__fields__]

# list of all fields, useful when exporting all this nonsense to a table
TABULAR_TEXT_HEADER = list(
    dict.fromkeys(TabularTEXT31.header() + TabularTEXT30.header())
)

#
# Parameters
#
# It is often useful to think of parameters either at the keyword level or at
# the "grouped" level (eg with all parameter keywords sharing the same index).
# This supplies a namedtuple for the former and a pydantic parser for the latter.
#
# Note that the pydantic parser is totally independent of the TEXT parser above.
#


class Ptype(StrEnum):
    BITS = "B"
    AMPTYPE = "E"
    MAXRANGE = "R"
    NAME = "N"
    FILTER = "F"
    GAIN = "G"
    WAVELENGTH = "L"
    POWER = "O"
    PERCENT_EMIT = "P"
    LONGNAME = "S"
    DETECTOR_TYPE = "T"
    VOLTAGE = "V"
    # 3.1 only
    CALIBRATION = "CALIBRATION"
    DISPLAY = "D"

    @classmethod
    def values30(self) -> set[Self]:
        return {x for x in self if x not in [Ptype.CALIBRATION, Ptype.DISPLAY]}

    @classmethod
    def values31(self) -> set[Self]:
        return {x for x in self}

    @classmethod
    def str_values30(self) -> set[str]:
        return {str(x) for x in self.values30()}

    @classmethod
    def str_values31(self) -> set[str]:
        return {str(x) for x in self.values31()}

    @classmethod
    def _build_param_re(self, xs: set[str]) -> re.Pattern[str]:
        # ASSUME this will match a standard key which is in uppercase
        return re.compile(f"^\\$P([0-9]+)({"|".join(xs)})$")

    @classmethod
    def re30(self) -> re.Pattern[str]:
        return self._build_param_re(self.str_values30())

    @classmethod
    def re31(self) -> re.Pattern[str]:
        return self._build_param_re(self.str_values31())


PARAM_RE_3_0 = Ptype.re30()
PARAM_RE_3_1 = Ptype.re31()


class AmpType(NamedTuple):
    """Decomposed type for the $PnE keyword"""

    decades: float
    zero: float


class Calibration(NamedTuple):
    """Decomposed type for the $PnCalibration keyword"""

    value: float
    unit: str


class LinDisplay(NamedTuple):
    """Decomposed type for the linear version of the $PnD keyword."""

    lower: float
    upper: float


class LogDisplay(NamedTuple):
    """Decomposed type for the log version of the $PnD keyword."""

    decades: float
    offset: float


Display = LinDisplay | LogDisplay


class ParsedParam(BaseModel):
    """
    One parameter denoted by a group of "$PnX" keywords with the same index.

    Methods are lowercased names of "X" in "$PnX"
    """

    b: int
    e: AmpType
    r: int
    n: str
    f: str | None
    g: float | None
    l: list[int] = []
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
        m = re.match("([0-9.]+), *([0-9.]+)", v)
        assert m is not None, f"amp type must be like 'f1,f2, got {v}'"
        return AmpType(float(m[1]), float(m[2]))

    @validator("l", pre=True)
    def validate_wavelength(cls, v: Any) -> list[int]:
        if v is None:
            return []
        assert isinstance(v, str), "wavelength must be string"
        if v == "NA":
            return []
        try:
            return [int(x) for x in v.split(",")]
        except ValueError:
            assert False, f"wavelength must be like 'n1[[,n2]...], got {v}'"

    @validator("v", pre=True)
    def validate_voltage(cls, v: Any) -> float | None:
        if v is None:
            return None
        assert isinstance(v, str), "voltage must be string"
        if v == "NA":
            return None
        try:
            return float(v)
        except ValueError:
            assert False, f"wavelength must be a float, got {v}'"

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

    @property
    def _amp_type_line(self) -> list[str]:
        e = self.e
        return [str(e.decades), str(e.zero)] if e is not None else ["", ""]

    @property
    def _calibration_line(self) -> list[str]:
        c = self.calibration
        return [str(c.value), c.unit] if c is not None else ["", ""]

    @property
    def _display_line(self) -> list[str]:
        d = self.d
        if isinstance(d, LinDisplay):
            return ["lin", str(d.lower), str(d.upper)]
        elif isinstance(d, LogDisplay):
            return ["log", str(d.decades), str(d.offset)]
        elif d is None:
            return [""] * 3
        else:
            assert_never(d)

    # useful for exporting to tables
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

    # useful for exporting to tables
    @property
    def line(self) -> list[str]:
        xs = [
            self.b,
            *self._amp_type_line,
            self.r,
            self.n,
            self.f,
            self.g,
            ",".join([str(x) for x in self.l]),
            self.o,
            self.p,
            self.t,
            self.v,
            *self._calibration_line,
            *self._display_line,
        ]
        return ["" if x is None else str(x) for x in xs]

    # TODO make a method here to "reverse" the grouping function and get
    # individual keywords back as a list


class ParamKeyword(NamedTuple):
    """A keyword describing a parameter (eg "$PnN")"""

    index_: ParamIndex
    ptype: Ptype
    value: str

    @property
    def key(self) -> str:
        return f"$P{self.index_}{self.ptype}"

    def serialize(self, delim: Delim) -> bytes:
        d = delim.to_bytes(1)
        return self.key.encode() + d + str(self.value).encode()


def group_params(ps: list[ParamKeyword]) -> dict[ParamIndex, ParsedParam]:
    """Group parameters by their index."""
    acc: dict[ParamIndex, dict[str, str]] = {}
    for p in ps:
        if p.index_ not in acc:
            acc[p.index_] = {}
        acc[p.index_][p.ptype.lower()] = p.value
    return {k: ParsedParam.parse_obj(v) for k, v in acc.items()}


#
# Reading FCS files
#


@dataclass(frozen=True)
class ParsedTEXT:
    """Represents the result of reading the TEXT segment."""

    standard: TabularTEXT
    params: list[ParamKeyword]
    deviant: dict[StandardKey, str]
    nonstandard: dict[str, str]


@dataclass(frozen=True)
class ParsedMetadata:
    """Represents the result of reading the metadata (ASCII) section."""

    header: FCSHeader
    meta: ParsedTEXT
    warnings: list[str]


@dataclass(frozen=True)
class ParsedEvents(ParsedMetadata):
    """Represents the result of reading the metadata and events."""

    events: pd.DataFrame


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


RM_PAT = re.compile("\\$(G|R|PK|PKN)[0-9]+.")


def split_meta(meta: dict[str, Any]) -> tuple[FCSHeader, ParsedTEXT]:
    # based on version, choose the parameter pattern and standard fields we need
    header, version = read_header(meta)
    pat = version.choose(PARAM_RE_3_0, PARAM_RE_3_1)
    standard_fields = version.choose(StandardKeys30, StandardKeys31)
    # remove header, convert all keys to uppercase (keys are case-insensitive),
    # and convert all values to strings and clean whitespace
    text = {
        kbig: str(v).strip()
        for k, v in meta.items()
        if not (k == "__header__" or re.match(RM_PAT, (kbig := k.upper())) is not None)
    }
    dollar = {StandardKey(k): v for k, v in text.items() if k.startswith("$")}
    nonstandard = {k: v for k, v in text.items() if not k.startswith("$")}
    # pull all parameters into their own class (these are special)
    params = {
        k: ParamKeyword(ParamIndex(int(m[1])), Ptype(m[2]), v)
        for k, v in dollar.items()
        if (m := re.match(pat, k)) is not None
    }
    nonparams = {k: v for k, v in dollar.items() if k not in params}
    # not all machines follow the real standard; they are supposed to put "$" in
    # front of all standard names, but this doesn't always happen :(
    parsed_standard = version.choose(TabularTEXT30, TabularTEXT31)(
        **{k[1:].lower(): v for k, v in nonparams.items() if k in standard_fields}
    )
    deviant = {k: v for k, v in nonparams.items() if k not in standard_fields}
    return header, ParsedTEXT(parsed_standard, [*params.values()], deviant, nonstandard)


def read_fcs_metadata(p: Path) -> ParsedMetadata:
    with warnings.catch_warnings(record=True, action="always") as w:
        meta = fp.parse(p, meta_data_only=True)
        warn_msgs = [str(x.message).replace("\n", " ") for x in w]
    header, _meta = split_meta(meta)
    return ParsedMetadata(header, _meta, warn_msgs)


def read_fcs(p: Path) -> ParsedEvents:
    with warnings.catch_warnings(record=True, action="always") as w:
        meta, data = fp.parse(p, channel_naming="$PnN")
        warn_msgs = [str(x.message).replace("\n", " ") for x in w]
    header, _meta = split_meta(meta)
    return ParsedEvents(header, _meta, warn_msgs, data)


#
# Writing FCS files
#


class WritableFCS(NamedTuple):
    """Represents data to write to an FCS file.

    Offsets will be calculated on the fly when writen and thus are not
    required/supplied via header or TEXT.
    """

    text: SerializableTEXT
    params: list[ParamKeyword]
    other: dict[str, str]
    events: pd.DataFrame


def serialize_parameters(ps: list[ParamKeyword], delim: Delim) -> bytes:
    d = delim.to_bytes(1)
    return d.join(p.serialize(delim) for p in ps)


def serialize_keywords(xs: dict[str, str], delim: Delim) -> bytes:
    d = delim.to_bytes(1)
    return d.join([k.encode() + d + v.encode() for k, v in xs.items()])


def serialize_header_and_data_offsets(
    ver: Version,
    nondataoffset_text_length: int,
    data_length: int,
    field_width: int,
    delim: Delim,
) -> bytes:
    xdelim = delim.to_bytes(1)
    # the beginning data offset can be computed based on the length of the
    # HEADER (which we can choose) plus the predicted length of the offset
    # fields themselves plus the length of the rest of the TEXT segment
    dataoffset_text_length = (
        _BEGINDATA_KEY_LENGTH + _ENDDATA_KEY_LENGTH + 5 + field_width * 2
    )
    begindata_offset = (
        HEADER_LENGTH + dataoffset_text_length + nondataoffset_text_length
    )
    # the end data offset is simply the beginning offset + the number of events
    # times the width of the events (which are assumed constant)
    enddata_offset = begindata_offset + data_length
    # The header only contains the TEXT offsets, since we assume that some FCS
    # files will be large enough that they will exceed the number of allowed
    # digits in the header.
    header = (
        f"FCS{ver.value}    {HEADER_LENGTH:>8}{begindata_offset - 1:>8}" + EMPTY * 4
    )
    # format the offsets, left-padding the numbers with 0's
    begindata = f"{begindata_offset:0>{field_width}}".encode()
    enddata = f"{enddata_offset:0>{field_width}}".encode()
    s = xdelim.join([_BEGINDATA_KEY, begindata, _ENDDATA_KEY, enddata])
    return header.encode() + xdelim + s + xdelim


def write_fcs(
    path: Path,
    p: WritableFCS,
    delim: Delim,
    double: bool,
    dataoffset_field_width: int,
) -> None:
    xdelim = delim.to_bytes(1)
    nrow, ncol = p.events.shape
    binary_format, event_width, datatype = (
        (f"<{ncol}d", 8, Datatype.DOUBLE)
        if double
        else (f"<{ncol}f", 4, Datatype.FLOAT)
    )

    required = TEXTMeta(
        beginanalysis=0,
        endanalysis=0,
        beginstext=0,
        endstext=0,
        byteord=Byteord.LITTLE,
        datatype=datatype,
        mode=Mode.LIST,
        nextdata=0,
        par=ncol,
        tot=nrow,
    )

    # ensure bits are properly set for each field
    new_params = [
        (
            ParamKeyword(p.index_, Ptype.BITS, str(event_width * 8))
            if p.ptype is Ptype.BITS
            else p
        )
        for p in p.params
    ]

    new_text = xdelim.join(
        x
        for x in [
            required.serialize(delim),
            p.text.serialize(delim),
            serialize_parameters(new_params, delim),
            serialize_keywords(p.other, delim),
        ]
        if len(x) > 0
    )

    # TODO if we are going to force 3.1 then we also technically should convert
    # some 3.0 keywords to 3.1 (COMP -> SPILLOVER for instance)
    header_dataoffsets = serialize_header_and_data_offsets(
        Version.v3_1,
        len(new_text),
        p.events.size * event_width,
        dataoffset_field_width,
        delim,
    )

    with open(path, "wb") as f:
        # write new HEADER+DATASTART/END
        f.write(header_dataoffsets)
        # write the rest of the TEXT keywords
        f.write(new_text)
        # write all the data
        for r in p.events.itertuples(name=None, index=False):
            f.write(struct.pack(binary_format, *r))


def with_fcs(
    ip: Path,
    op: Path,
    fun: Callable[[ParsedEvents], WritableFCS],
    delim: Delim,
    double: bool,
    dataoffset_field_width: int,
) -> None:
    p = read_fcs(ip)
    write_fcs(op, fun(p), delim, double, dataoffset_field_width)

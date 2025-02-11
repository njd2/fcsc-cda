import pandas as pd
import re
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import NamedTuple, NewType, Self, Generic, TypeVar, assert_never

FileIndex = NewType("FileIndex", int)
OM = NewType("OM", str)
Rep = NewType("Rep", int)
Lot = NewType("Lot", int)


NO_SCATTER = [
    OM("LMNXSEA_CellStream-1"),
    OM("LMNXSEA_CellStream-2"),
    OM("LMNXSEA_ImageStreamX-1"),
]


class Matrix(Enum):
    Matrix1 = 1
    Matrix2 = 2
    Matrix3 = 3

    @property
    def name(self) -> str:
        return f"Matrix {self.value}"


class FMOMarker(Enum):
    CD14 = "CD14"
    CD19 = "CD19"


class Machine(NamedTuple):
    org: str
    machine: str

    @classmethod
    def tsv_header(cls) -> list[str]:
        return ["org", "machine", "om"]

    @property
    def line(self) -> list[str]:
        return [self.org, self.machine, self.om]

    @property
    def om(self) -> OM:
        return OM(f"{self.org}_{self.machine}")


class Color(Enum):
    V450 = "v450"
    V500 = "v500"
    FITC = "fitc"
    PC55 = "pc55"
    PE = "pe"
    PC7 = "pc7"
    APC = "apc"
    AC7 = "ac7"

    @classmethod
    def from_long_name(cls, s: str) -> "Color":
        if s == "V450":
            return Color.V450
        elif s == "V500-C":
            return Color.V500
        elif s == "FITC":
            return Color.FITC
        elif s == "PerCP-Cy5.5":
            return Color.PC55
        elif s == "PE":
            return Color.PE
        elif s == "PE-Cy7":
            return Color.PC7
        elif s == "APC":
            return Color.APC
        elif s == "APC-Cy7":
            return Color.AC7
        else:
            raise ValueError(f"invalid long name: {s}")

    @classmethod
    def from_livedead(cls, s: str) -> "Color":
        if s == "AxPB":
            # annexin V
            return Color.V450
        elif s == "AqLD":
            # aqua live/dead
            return Color.V500
        else:
            # these should be the only stains in the whole dataset
            raise ValueError(f"invalid live/dead name: {s}")

    @classmethod
    def from_compbead(cls, s: str) -> "Color":
        if s == "PBBead":
            # Pacific Blue Bead
            return Color.V450
        elif s == "AqBead":
            # Aqua Live/Dead Bead
            return Color.V500
        else:
            # There should only be 2 beads in the entire dataset
            raise ValueError(f"invalid compensation bead name: {s}")

    @property
    def index(self) -> int:
        return [*Color].index(self)

    def __lt__(self, other: Self) -> bool:
        return self.index < other.index

    def __gt__(self, other: Self) -> bool:
        return self.index > other.index

    def __le__(self, other: Self) -> bool:
        return self == other or self < other

    def __ge__(self, other: Self) -> bool:
        return self == other or self > other


class CompMaterial(Enum):
    PBMC = 0
    LYOLEUK = 2
    VERSA = 3
    ABC = 4
    COMP = 5

    @property
    def matrix(self) -> Matrix | None:
        if self is CompMaterial.PBMC or self is CompMaterial.LYOLEUK:
            return Matrix.Matrix1
        elif self is CompMaterial.ABC:
            return Matrix.Matrix2
        elif self is CompMaterial.VERSA:
            return Matrix.Matrix3
        elif self is CompMaterial.COMP:
            return None
        else:
            assert_never(self)


_SOP1_COLORS = [
    "V450",
    "V500-C",
    "FITC",
    "PerCP-Cy5.5",
    "PE",
    "PE-Cy7",
    "APC",
    "APC-Cy7",
]

_SOP1_COLORMAP = {f"FC-{k}": v for k, v in zip(_SOP1_COLORS, Color)}


_SOP2_MATERIAL_MAP = {
    "Versacmp": CompMaterial.VERSA,
    "AbCTotal": CompMaterial.ABC,
    "lyoLeuk": CompMaterial.LYOLEUK,
    "cryoPBMC": CompMaterial.PBMC,
}


def parse_bead_cell_type(s: str) -> CompMaterial:
    if s == "Versacmp":
        return CompMaterial.VERSA
    elif s == "AbCTotal":
        return CompMaterial.ABC
    elif s == "lyoLeuk":
        return CompMaterial.LYOLEUK
    elif s == "cryoPBMC":
        return CompMaterial.PBMC
    else:
        # compensation beads don't have an indicator in the material string
        raise ValueError(f"Invalid bead/cell type: {s}")


def parse_sop2_material(s: str) -> tuple[Color, CompMaterial, Matrix | None]:
    try:
        return (Color.from_compbead(s), CompMaterial.COMP, None)
    except ValueError:
        if (m := re.match("^(.+)-([^-]+?)$", s)) is None:
            raise ValueError(f"Could not parse SOP2 material {s}")
        mat = parse_bead_cell_type(m[2])
        f = Color.from_livedead if mat is CompMaterial.LYOLEUK else Color.from_long_name
        return (f(m[1]), mat, mat.matrix)


_SOP3_FMO_COLORMAP = {
    f"cryoPBMC-{k}-fmo": v
    for k, v in {
        "AxPB": (Color.V450, None),
        "AqLD": (Color.V500, None),
        "FITC": (Color.FITC, None),
        "PerCP-Cy5.5": (Color.PC55, None),
        "PE": (Color.PE, None),
        "PE-Cy7": (Color.PC7, None),
        "APC-CD14": (Color.APC, None),
        "APC-CD19": (Color.APC, None),
        "APC-Cy7": (Color.AC7, None),
    }.items()
}


# TODO this has an analogue in R, not DRY
def format_group(sop: int, exp: int, material: str) -> str:
    if sop == 1:
        return "SOP 1"
    elif sop == 2:
        n = "3/4" if exp == 4 else str(exp)
        return f"SOP 2: Matrix {n}"
    else:
        if "fmo" in material:
            s = "Test FMO"
        elif exp == 1:
            s = "Test Pheno"
        elif exp == 2:
            s = "Test Count"
        elif exp == 3:
            s = "QC Count"
        else:
            s = "QC Pheno"
        return f"SOP 3: {s}"


RangeMap = dict[FileIndex, dict[Color, int]]


def read_range_map(params: Path) -> RangeMap:
    df = pd.read_table(
        params,
        usecols=["file_index", "maxrange", "shortname"],
    )
    acc: RangeMap = {}
    for file_index, maxrange, shortname in df.itertuples(index=False):
        try:
            color = Color(shortname)
        except ValueError:
            color = None
        if color is not None:
            p = FileIndex(int(file_index))
            if p not in acc:
                acc[p] = {}
            acc[p][color] = int(maxrange)
    return acc


class CalibrationMeta(NamedTuple):
    machine: Machine
    color: Color | None


class CompensationMeta(NamedTuple):
    machine: Machine
    color: Color
    matrix: Matrix | None
    material: CompMaterial


class FMOMeta(NamedTuple):
    machine: Machine
    color: Color
    marker: FMOMarker | None


class PhenoMeta(NamedTuple):
    machine: Machine
    lot: Lot
    rep: Rep


class CountMeta(NamedTuple):
    machine: Machine
    lot: Lot
    rep: Rep


class QCPhenoMeta(NamedTuple):
    machine: Machine
    rep: Rep


class QCCountMeta(NamedTuple):
    machine: Machine
    rep: Rep


class IndexedPath(NamedTuple):
    file_index: FileIndex
    filepath: Path

    @classmethod
    def tsv_header(cls) -> list[str]:
        return ["file_index", "filepath"]

    @property
    def line(self) -> list[str]:
        return [str(self.file_index), str(self.filepath)]


M = TypeVar(
    "M",
    CalibrationMeta,
    CompensationMeta,
    FMOMeta,
    PhenoMeta,
    CountMeta,
    QCPhenoMeta,
    QCCountMeta,
)


@dataclass(frozen=True)
class SplitFCSMeta(Generic[M]):
    indexed_path: IndexedPath
    filemeta: M


AnyMeta = (
    CalibrationMeta
    | CompensationMeta
    | FMOMeta
    | PhenoMeta
    | CountMeta
    | QCPhenoMeta
    | QCCountMeta
)

FcsCalibrationMeta = SplitFCSMeta[CalibrationMeta]
FcsCompensationMeta = SplitFCSMeta[CompensationMeta]
FcsFMOMeta = SplitFCSMeta[FMOMeta]
FcsPhenoMeta = SplitFCSMeta[PhenoMeta]
FcsCountMeta = SplitFCSMeta[CountMeta]
FcsQCPhenoMeta = SplitFCSMeta[QCPhenoMeta]
FcsQCCountMeta = SplitFCSMeta[QCCountMeta]

AnySplitFCSMeta = (
    FcsCalibrationMeta
    | FcsCompensationMeta
    | FcsFMOMeta
    | FcsPhenoMeta
    | FcsCountMeta
    | FcsQCPhenoMeta
    | FcsQCCountMeta
)


class FCSPathMeta(NamedTuple):
    machine: Machine
    material: str
    sop: int
    eid: int
    rep: Rep

    @classmethod
    def tsv_header(cls) -> list[str]:
        return [*Machine.tsv_header(), *FCSPathMeta._fields[1:], "group"]

    @property
    def line(self) -> list[str]:
        return [
            *self.machine.line,
            *[str(x) for x in self._asdict().values()][1:],
            self.group,
        ]

    @property
    def group(self) -> str:
        return format_group(self.sop, self.eid, self.material)

    @property
    def projection(self) -> AnyMeta:
        if self.sop == 1:
            try:
                color = (
                    None
                    if self.material == "RCP-30-5A"
                    else _SOP1_COLORMAP[self.material]
                )
                return CalibrationMeta(self.machine, color)
            except KeyError:
                raise ValueError(f"invalid color for SOP 1: {self}")

        elif self.sop == 2:
            color, mat, matrix = parse_sop2_material(self.material)
            if not (
                (self.eid == 1 and matrix is Matrix.Matrix1)
                or (self.eid == 2 and matrix is Matrix.Matrix2)
                or (self.eid == 3 and matrix is Matrix.Matrix3)
                or (self.eid == 4 and matrix is None)
            ):
                raise ValueError(f"invalid color for SOP 2: {self}")

            return CompensationMeta(self.machine, color, matrix, mat)

        elif self.sop == 3:
            if self.eid == 1:
                try:
                    color, marker = _SOP3_FMO_COLORMAP[self.material]
                    return FMOMeta(self.machine, color, marker)
                except KeyError:
                    m = re.match("panel1-cryoPBMC-([1-3])", self.material)
                    if m is not None and 1 <= self.rep <= 3:
                        lot = Lot(int(m[1]))
                        return PhenoMeta(self.machine, lot, self.rep)
                    else:
                        raise ValueError(f"invalid SOP 3, EID 1: {self}")
            elif self.eid == 2:
                m = re.match("AqLD-TruCount-CD45-AxPB-cryoPBMC-([1-3])", self.material)
                if m is not None and 1 <= self.rep <= 3:
                    lot = Lot(int(m[1]))
                    return CountMeta(self.machine, lot, self.rep)
                else:
                    raise ValueError(f"invalid SOP 3, EID 2: {self}")
            elif (
                self.eid == 3
                and self.material == "TruCount-CD45-TruCytes"
                and 1 <= self.rep <= 2
            ):
                return QCCountMeta(self.machine, self.rep)
            elif (
                self.eid == 4
                and self.material == "panel1-TruCytes"
                and 1 <= self.rep <= 3
            ):
                return QCPhenoMeta(self.machine, self.rep)
            else:
                raise ValueError(f"invalid SOP 3: {self}")

        else:
            raise ValueError(f"invalid: {self}")


class FCSMeta(NamedTuple):
    indexed_path: IndexedPath
    filemeta: FCSPathMeta

    @classmethod
    def tsv_header(cls) -> list[str]:
        return [*IndexedPath.tsv_header(), *FCSPathMeta.tsv_header()]

    @property
    def line(self) -> list[str]:
        return [*self.indexed_path.line, *self.filemeta.line]


# class SplitFCSMeta(NamedTuple):
#     indexed_path: IndexedPath
#     filemeta: SplitFileMeta


def split_path(p: Path) -> FCSPathMeta:
    """Split an FCS path into its components."""
    s = p.name.split("_")
    try:
        return FCSPathMeta(
            machine=Machine(s[2], s[3]),
            material=s[4],
            sop=int(s[5][5]),
            eid=int(s[6][1]),
            rep=Rep(int(s[8])),
        )
    except IndexError:
        raise ValueError(f"not a valid FCS path: {p}")


def split_indexed_path(p: IndexedPath) -> AnySplitFCSMeta:
    s = split_path(p.filepath).projection
    # TODO mypy doesn't understand that a union inside a tuple can be lifted
    if isinstance(s, CalibrationMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, CompensationMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, FMOMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, PhenoMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, CountMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, QCCountMeta):
        return SplitFCSMeta(p, s)
    elif isinstance(s, QCPhenoMeta):
        return SplitFCSMeta(p, s)
    else:
        assert_never(s)


def read_files(files: Path) -> list[AnySplitFCSMeta]:
    """Read a tsv like "index, filepath" and return parsed metadata."""

    df = pd.read_table(files, names=["file_index", "file_path"])
    return [
        split_indexed_path(IndexedPath(FileIndex(int(i)), Path(p)))
        for i, p in df.itertuples(index=False)
    ]

import pandas as pd
import re
from pathlib import Path
from enum import Enum
from typing import NamedTuple, NewType, Self

FileIndex = NewType("FileIndex", int)
OM = NewType("OM", str)
Rep = NewType("Rep", int)
Lot = NewType("Lot", int)


class Matrix(Enum):
    Matrix1 = 0
    Matrix2 = 1
    Matrix3 = 2


class FMOMarker(Enum):
    CD14 = "CD14"
    CD19 = "CD19"


class Machine(NamedTuple):
    org: str
    machine: str

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

_SOP2_M1_MATERIALS = [
    "AxPB-lyoLeuk",
    "AqLD-lyoLeuk",
    "FITC-cryoPBMC",
    "PerCP-Cy5.5-cryoPBMC",
    "PE-cryoPBMC",
    "PE-Cy7-cryoPBMC",
    "APC-cryoPBMC",
    "APC-Cy7-cryoPBMC",
]


_SOP2_M2_MATERIALS = [
    "FITC-AbCTotal",
    "PerCP-Cy5.5-AbCTotal",
    "PE-AbCTotal",
    "PE-Cy7-AbCTotal",
    "APC-Cy7-AbCTotal",
    "APC-AbCTotal",
]

_SOP2_M3_MATERIALS = [
    "FITC-Versacmp",
    "PerCP-Cy5.5-Versacmp",
    "PE-Versacmp",
    "PE-Cy7-Versacmp",
    "APC-Versacmp",
    "APC-Cy7-Versacmp",
]

_SOP2_M4_MATERIALS = ["PBBead", "AqBead"]

_SOP2_M1_COLORMAP = {k: v for k, v in zip(_SOP2_M1_MATERIALS, Color)}
_SOP2_M2_COLORMAP = {k: v for k, v in zip(_SOP2_M2_MATERIALS, [*Color][2:])}
_SOP2_M3_COLORMAP = {k: v for k, v in zip(_SOP2_M3_MATERIALS, [*Color][2:])}
_SOP2_M4_COLORMAP = {k: v for k, v in zip(_SOP2_M4_MATERIALS, [*Color][:2])}


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


class CalibrationMeta(NamedTuple):
    machine: Machine
    color: Color | None


class CompensationMeta(NamedTuple):
    machine: Machine
    color: Color
    matrix: Matrix | None


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


SplitFileMeta = (
    CalibrationMeta
    | CompensationMeta
    | FMOMeta
    | PhenoMeta
    | CountMeta
    | QCPhenoMeta
    | QCCountMeta
)


class FCSPathMeta(NamedTuple):
    machine: Machine
    material: str
    sop: int
    eid: int
    rep: Rep

    @property
    def group(self) -> str:
        return format_group(self.sop, self.eid, self.material)

    @property
    def projection(self) -> SplitFileMeta:
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
            if self.eid == 1 and self.material in _SOP2_M1_COLORMAP:
                cm = _SOP2_M1_COLORMAP
                mat = Matrix.Matrix1
            elif self.eid == 2 and self.material in _SOP2_M2_COLORMAP:
                cm = _SOP2_M2_COLORMAP
                mat = Matrix.Matrix2
            elif self.eid == 3 and self.material in _SOP2_M3_COLORMAP:
                cm = _SOP2_M3_COLORMAP
                mat = Matrix.Matrix3
            elif self.eid == 4 and self.material in _SOP2_M4_COLORMAP:
                cm = _SOP2_M4_COLORMAP
                mat = None
            else:
                raise ValueError(f"invalid SOP 2: {self}")
            return CompensationMeta(self.machine, cm[self.material], mat)

        elif self.sop == 3:
            if self.eid == 1:
                try:
                    color, marker = _SOP3_FMO_COLORMAP[self.material]
                    return FMOMeta(self.machine, color, marker)
                except KeyError:
                    m = re.match("panel1-cryoPBMC-([1-3])", self.material)
                    if m is not None and 1 <= self.rep <= 3:
                        lot = Lot(int(m[0]))
                        return PhenoMeta(self.machine, lot, self.rep)
                    else:
                        raise ValueError(f"invalid SOP 3, EID 1: {self}")
            elif self.eid == 2:
                m = re.match("AqLD-TruCount-CD45-AxPB-cryoPBMC-([1-3])", self.material)
                if m is not None and 1 <= self.rep <= 3:
                    lot = Lot(int(m[0]))
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


class IndexedPath(NamedTuple):
    file_index: FileIndex
    filepath: Path


class FCSMeta(NamedTuple):
    indexed_path: IndexedPath
    filemeta: FCSPathMeta

    @classmethod
    def tsv_header(cls) -> list[str]:
        return ["file_path", *FCSPathMeta._fields, "filepath"]

    @property
    def line(self) -> list[str]:
        return [
            str(self.indexed_path.file_index),
            self.filemeta.machine.org,
            self.filemeta.machine.machine,
            self.filemeta.material,
            str(self.filemeta.sop),
            str(self.filemeta.eid),
            str(self.filemeta.rep),
            self.filemeta.group,
            self.filemeta.machine.om,
            str(self.indexed_path.filepath),
        ]


class SplitFCSMeta(NamedTuple):
    indexed_path: IndexedPath
    filemeta: SplitFileMeta


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


def read_files(files: Path) -> list[SplitFCSMeta]:
    """Read a tsv like "index, filepath" and return parsed metadata."""

    df = pd.read_table(files, names=["file_index", "file_path"])
    return [
        SplitFCSMeta(
            IndexedPath(FileIndex(int(i)), _p := Path(p)),
            split_path(_p).projection,
        )
        for i, p in df.itertuples(index=False)
    ]

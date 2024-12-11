import pandas as pd
from pathlib import Path
from typing import NamedTuple, Any
from multiprocessing import Pool
from common.io import (
    Ptype,
    Delim,
    WritableFCS,
    ParsedEvents,
    with_fcs,
    ParamKeyword,
    ParamIndex,
)

# Convert FCS files to standardized format including:
# - standardizing channel names
# - removing extra channels
# - exporting to FCS3.1 format
# - saving all data as 32-bit floats
# - removing compensation matrices (which should be meaningless at this stage)


# these take lots of space and shouldn't be necessary to do anything important
EXCLUDED_FIELDS = {"comp", "spillover", "unicode"}


class ChannelNames(NamedTuple):
    param_index: ParamIndex
    short: str
    long: str

    def _to_param(self, name: Ptype, value: str) -> ParamKeyword:
        return ParamKeyword(self.param_index, name, value)

    def to_bits(self, bits: int) -> ParamKeyword:
        return self._to_param(Ptype.BITS, str(bits))

    @property
    def to_short(self) -> ParamKeyword:
        return self._to_param(Ptype.NAME, self.short)

    @property
    def to_long(self) -> ParamKeyword:
        return self._to_param(Ptype.LONGNAME, self.long)


class PreChannelNames(NamedTuple):
    short: str
    long: str

    def add_index(self, i: int) -> ChannelNames:
        return ChannelNames(ParamIndex(i), self.short, self.long)


ChannelMap = dict[str, ChannelNames]
PreChannelMap = dict[str, PreChannelNames]
OrgMachChannelMap = dict[tuple[str, str], PreChannelMap]
TextKWs = dict[str, str]


class RunConfig(NamedTuple):
    ipath: Path
    opath: Path
    sop: int
    eid: int
    missing_scatter_height: bool
    channel_map: PreChannelMap
    start: int
    end: int


def format_parameters(
    ps: list[ParamKeyword], cm: ChannelMap, bits: int
) -> list[ParamKeyword]:
    # build a mapping between the current indices and the new indices/names; note
    # that that ChannelMap should have a mapping between the $PnN value and
    # the standardized index, $PnN, and $PnS values
    index_map: dict[ParamIndex, ChannelNames] = {
        p.index_: cm[p.value]
        for p in ps
        if isinstance(p.value, str) and p.value in cm and p.ptype == "N"
    }
    # reindex all parameters except for $PnS, $PnN, and $PnB
    reindexed = [
        ParamKeyword(index_map[p.index_].param_index, p.ptype, p.value)
        for p in ps
        if p.index_ in index_map and p.ptype not in "NSB"
    ]
    # rebuild the $PnS, $PnN, and $PnB parameters (some of these might not be
    # present so easier to build from scratch and add rather than selectively
    # replace/add). In the case of $PnB, rebuilding rather than reusing
    # guarantees all outputs will be in the same format
    rebuilt = [
        y
        for x in index_map.values()
        for y in [
            x.to_short,
            x.to_long,
            x.to_bits(bits),
        ]
    ]
    # sort by index/type and serialize
    new_params = reindexed + rebuilt
    new_params.sort(key=lambda x: (x.index_, x.ptype))
    return new_params


def standardize_fcs(c: RunConfig) -> None:
    def go(p: ParsedEvents) -> WritableFCS:
        # remove height channels if this file used beads, or if they are missing
        # from the FCS file (since this isn't fatal)
        if c.sop == 1 or c.sop == 2 and c.eid != 1 or c.missing_scatter_height:
            pm = {
                k: v
                for k, v in c.channel_map.items()
                if v.short not in ["ssc_h", "fsc_h"]
            }
        else:
            pm = c.channel_map
        m = {k: v.add_index(i + 1) for i, (k, v) in enumerate(pm.items())}
        new_df = p.events[c.start : c.end + 1][[*m]]
        new_params = format_parameters(p.meta.params, m, 32)
        other = {**{str(k): v for k, v in p.meta.deviant.items()}, **p.meta.nonstandard}
        meta = p.meta.standard.serializable(EXCLUDED_FIELDS)
        return WritableFCS(meta, new_params, other, new_df)

    with_fcs(c.ipath, c.opath, go, Delim(30), False, 12)


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
            # Missing machine names mean that this channel has no name, which
            # is the case for the scatter channels of imaging cytometer. These
            # should be removed since there is no channel to export.
            if machine_name == "NA":
                continue
            std_name = s[3].strip()
            std_name_long = s[4].strip()
            key = (org, machine)
            if key not in acc:
                acc[key] = {}
            acc[key][machine_name] = PreChannelNames(std_name, std_name_long)
    return acc


def main(smk: Any) -> None:
    top_gates_in = Path(smk.input["top"])
    channels_in = Path(smk.input["channels"])
    meta_in = Path(smk.input["meta"])

    flag_out = Path(smk.output[0])

    out_dir = flag_out.parent

    COLUMNS = [
        "org",
        "machine",
        "sop",
        "eid",
        "filepath",
        "start",
        "end",
        "missing_scatter_height",
    ]

    # Skip files for which the channel definitions are not complete. We also
    # skip files that don't have "height" scatter channels but whether this is
    # a real issue depends on the type of file, so deal with that using more
    # complex logic compared to this dumb filter.
    ISSUE_COLUMNS = ["missing_time", "missing_colors", "missing_scatter_area"]

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
            int(sop),
            int(eid),
            mi_sc_hght,
            om_channel_map[org, machine],
            start,
            end,
        )
        for org, machine, sop, eid, filepath, start, end, mi_sc_hght in df.itertuples(
            index=False
        )
    ]

    with Pool(smk.threads) as p:
        p.map(standardize_fcs, runs)
        # list(map(standardize_fcs, [runs[0]]))

    flag_out.touch()


main(snakemake)  # type: ignore

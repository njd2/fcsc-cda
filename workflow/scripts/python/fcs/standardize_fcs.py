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

# Convert FCS files into "standard format" (as defined by this pipeline) which
# includes filtering to "valid" events and choosing only the channels necessary
# for downstream analysis.


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


ChannelMap = dict[str, ChannelNames]
OrgMachChannelMap = dict[tuple[str, str], ChannelMap]
TextKWs = dict[str, str]


class RunConfig(NamedTuple):
    ipath: Path
    opath: Path
    channel_map: ChannelMap
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


EXCLUDED_FIELDS = {"comp", "spillover"}


def standardize_fcs(c: RunConfig) -> None:
    def go(p: ParsedEvents) -> WritableFCS:
        new_df = p.events[c.start : c.end + 1][[*c.channel_map]]
        new_params = format_parameters(p.meta.params, c.channel_map, 32)
        other = {**{str(k): v for k, v in p.meta.deviant.items()}, **p.meta.nonstandard}
        meta = p.meta.standard.serializable(EXCLUDED_FIELDS)
        return WritableFCS(meta, new_params, other, new_df)

    with_fcs(c.ipath, c.opath, go, Delim(30), False, 12)


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

import sys
import os
import numpy as np
import numpy.typing as npt
from pathlib import Path
import flowkit as fk  # type: ignore
from flowkit import Dimension, Sample, GatingStrategy
from flowkit._models.gates import RectangleGate  # type: ignore
from typing import NamedTuple
from multiprocessing import Pool
from common.io import read_fcs
import common.metadata as ma
import common.gating as ga
from common.functional import fmap_maybe, from_maybe


def build_gating_strategy(
    gs: ga.SOP1Gates,
    color_ranges: ma.RangeMap,
    fcs: ma.FcsCalibrationMeta,
    scatteronly: bool,
) -> tuple[GatingStrategy, Sample, npt.NDArray[np.bool_], ga.GatingStrategyDebug]:
    om = fcs.filemeta.machine.om
    pathname = fcs.indexed_path.filepath.name
    color = fcs.filemeta.color
    g_strat = GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    bead_gate = ga.build_scatter_gate(gs.scatter_gates[om].from_color(color))
    g_strat.add_gate(bead_gate, ("root",))

    # The color gates are automatically placed according to events, so read
    # events, make a flowkit Sample, then gate out the beads
    parsed = read_fcs(fcs.indexed_path.filepath)
    smp = Sample(parsed.events, sample_id=str(fcs.indexed_path.filepath.name))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    if scatteronly:
        return (g_strat, smp, mask, ga.GatingStrategyDebug(pathname, {}))

    # Apply logicle transform to each color channel.
    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    trans_conf = (
        gs.transform_configs.rainbow if color is None else gs.transform_configs.fc
    )
    trans = ga.transform_colors(
        trans_conf,
        df_beads,
        color_ranges[fcs.indexed_path.file_index],
    )

    for c, t in trans.items():
        g_strat.add_transform(c.logicle_id, t)

    smp.apply_transform({c.value: t for c, t in trans.items()})
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    # Place gates on each color channel. This will be different depending on if
    # these are rainbow beads or FC beads. The former should have 8 peaks in all
    # channels, and the latter should have exactly two peaks in its own color
    # channel. In the case of rainbow, only keep the gates for the color with
    # the most peaks (up to 8) and area under these peaks. In all cases, place
    # gate using peak/valley heuristic for finding "large spikes" in the bead
    # population. Do this on transformed data since this is the only sane way to
    # resolve the lower peaks.
    if color is None:
        # rainbow beads
        rs = [
            (
                c,
                ga.make_min_density_serial_gates(
                    gs.autogate_configs.rainbow,
                    df_beads_x[c.value].values,
                    8,
                ),
            )
            for c in ma.Color
        ]
        # return all results in case we want to debug them...
        gate_results = {k.value: v for k, v in dict(rs).items()}
        # ...but only keep the best color in terms of peak resolution
        # TODO use some metric for "peak separation" here, like the distance b/t
        # the quantiles relative to the distances b/t gates
        maxres = max(rs, key=lambda r: (len(r[1].final_intervals), r[1].final_area))
        gate_color = maxres[0]
        ints = maxres[1].xintervals
    else:
        # fc beads
        x = df_beads_x[color.value].values
        r = ga.make_min_density_serial_gates(gs.autogate_configs.fc, x, 2)
        gate_results = {color.value: r}
        gate_color = color
        ints = r.xintervals

    for i, s in enumerate(ints):
        g_strat.add_gate(gate_color.to_1d_gate(i, (s.x0, s.x1)), ("root", "beads"))
    return (
        g_strat,
        smp,
        mask,
        ga.GatingStrategyDebug(pathname, gate_results),
    )


def read_paths(files: Path) -> set[ma.FcsCalibrationMeta]:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    fs = ma.read_files(files)
    return set(
        ma.FcsCalibrationMeta(f.indexed_path, f.filemeta)
        for f in fs
        if f.filemeta.machine.om not in ma.NO_SCATTER
        and isinstance(f.filemeta, ma.CalibrationMeta)
    )


# def read_path_map(files: Path) -> FileMap:
#     """Read a tsv like "index, filepath" and return paths for SOP 1."""
#     calibrations = read_paths(files)
#     return {
#         om: set(x.indexed_path for x in gs)
#         for om, gs in groupby(calibrations, lambda c: c.filemeta.machine.om)
#     }


class CalibrationRun(NamedTuple):
    fcs: ma.FcsCalibrationMeta
    gate_config: ga.SOP1Gates
    color_ranges: ma.RangeMap
    out_dir: Path | None

    @property
    def out(self) -> Path | None:
        color = from_maybe("rainbow", self.fcs.filemeta.color)
        om = self.fcs.filemeta.machine.om
        fn = f"{om}-{color}.xml"
        return fmap_maybe(lambda p: p / fn, self.out_dir)


def write_gating_strategy(out: Path | None, gs: fk.GatingStrategy) -> None:
    if out is not None:
        with open(out, "wb") as f:
            fk.export_gatingml(gs, f)
    else:
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            fk.export_gatingml(gs, f)


def write_gate_inner(r: CalibrationRun) -> tuple[ma.IndexedPath, Path | None]:
    gs = build_gating_strategy(r.gate_config, r.color_ranges, r.fcs, False)[0]
    write_gating_strategy(r.out, gs)
    return (r.fcs.indexed_path, r.out)


# def write_gate(
#     om: ma.OM,
#     boundaries: Path,
#     params: Path,
#     fcs: ma.IndexedPath,
#     out: Path | None,
# ) -> tuple[ma.IndexedPath, Path | None]:
#     range_map = ma.read_range_map(params)
#     color_ranges = range_map[fcs.file_index]
#     return write_gate_inner(CalibrationRun(om, boundaries, color_ranges, fcs, out))


def write_all_gates(
    files: Path,
    boundaries: Path,
    params: Path,
    out_dir: Path | None,
    threads: int | None = None,
) -> list[tuple[ma.IndexedPath, Path | None]]:

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    calibrations = read_paths(files)

    gs = ga.read_gates(boundaries).sop1
    range_map = ma.read_range_map(params)

    runs = [CalibrationRun(c, gs, range_map, out_dir) for c in calibrations]

    if threads is None:
        return list(map(write_gate_inner, runs))
    else:
        with Pool(threads) as pl:
            return pl.map(write_gate_inner, runs)

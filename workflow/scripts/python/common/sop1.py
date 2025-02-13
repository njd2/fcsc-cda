import sys
import os
import numpy as np
import numpy.typing as npt
from itertools import groupby
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

LogicleM = 4.5

FileMap = dict[ma.OM, list[ma.IndexedPath]]
FileRangeMap = dict[ma.OM, dict[ma.IndexedPath, dict[ma.Color, int]]]
V_F32 = npt.NDArray[np.float32]


# None = rainbow
GateRangeMap = dict[ma.OM, dict[ma.Color | None, ga.AnyBounds]]


class D1Root(NamedTuple):
    """1st derivative roots of f, either a "peak" or "valley" for f."""

    x: float
    is_peak: bool


def path_to_color(p: Path) -> ma.Color | None:
    x = ma.split_path(p).projection
    assert isinstance(x, ma.CalibrationMeta)
    return x.color


def build_gating_strategy(
    sc: ga.SOP1Gates,
    gs: ga.AnyBounds,
    color_ranges: dict[ma.Color, int],
    fcs_path: Path,
    scatteronly: bool,
) -> tuple[GatingStrategy, Sample, npt.NDArray[np.bool_], ga.GatingStrategyDebug]:
    bead_color = path_to_color(fcs_path)
    g_strat = GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    bead_gate = ga.build_scatter_gate(gs)
    g_strat.add_gate(bead_gate, ("root",))

    # The color gates are automatically placed according to events, so read
    # events, make a flowkit Sample, then gate out the beads
    parsed = read_fcs(fcs_path)
    smp = Sample(parsed.events, sample_id=str(fcs_path.name))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    if scatteronly:
        return g_strat, smp, mask, ga.GatingStrategyDebug(fcs_path.name, {})

    # Apply logicle transform to each color channel. In this case there should
    # be relatively few events in the negative range, so A should be 0. Set M to
    # be 4.5 (sane default). I don't feel like getting the original range data
    # for each channel so just use the max of all of them (T = max). Then set W
    # according to 5% negative heuristic (see Parks et al (the Logicle Paper)
    # for formulas/rationale for doing this).
    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    trans = {}
    trans_f = (
        sc.transform_configs.rainbow.to_transform
        if bead_color is None
        else sc.transform_configs.fc.to_transform
    )
    for c in ma.Color:
        arr = df_beads[c.value].values
        maxrange = float(color_ranges[c])
        trans[c.value] = trans_f(arr, maxrange)

    for k, v in trans.items():
        g_strat.add_transform(f"{k}_logicle", v)

    smp.apply_transform(trans)
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    # Place gates on each color channel. This will be different depending on if
    # these are rainbow beads or FC beads. The former should have 8 peaks in all
    # channels, and the latter should have exactly two peaks in its own color
    # channel. In the case of rainbow, only keep the gates for the color with
    # the most peaks (up to 8) and area under these peaks. In all cases, place
    # gate using peak/valley heuristic for finding "large spikes" in the bead
    # population. Do this on transformed data since this is the only sane way to
    # resolve the lower peaks.
    gate_results = {}
    if bead_color is None:
        # rainbow beads
        rs = [
            (
                c,
                ga.make_min_density_serial_gates(
                    sc.autogate_configs.rainbow,
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
        gate_color = maxres[0].value
        ints = maxres[1].xintervals
    else:
        # fc beads
        x = df_beads_x[bead_color.value].values
        r = ga.make_min_density_serial_gates(sc.autogate_configs.fc, x, 2)
        gate_results[bead_color.value] = r
        gate_color = bead_color.value
        ints = r.xintervals

    gates = [
        RectangleGate(
            f"{gate_color}_{i}",
            [
                Dimension(
                    gate_color,
                    transformation_ref=f"{gate_color}_logicle",
                    range_min=s.x0,
                    range_max=s.x1,
                )
            ],
        )
        for i, s in enumerate(ints)
    ]
    for g in gates:
        g_strat.add_gate(g, ("root", "beads"))
    return g_strat, smp, mask, ga.GatingStrategyDebug(fcs_path.name, gate_results)


def read_path_map(files: Path) -> FileMap:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    fs = ma.read_files(files)
    calibrations = [
        (f.indexed_path, f.filemeta)
        for f in fs
        if f.filemeta.machine.om not in ma.NO_SCATTER
        and isinstance(f.filemeta, ma.CalibrationMeta)
    ]
    return {
        om: [
            x[0]
            for x in sorted(
                gs, key=lambda x: 9 if x[1].color is None else x[1].color.index
            )
        ]
        for om, gs in groupby(calibrations, lambda c: c[1].machine.om)
    }


def read_path_range_map(files: Path, params: Path) -> FileRangeMap:
    path_map = read_path_map(files)
    range_map = ma.read_range_map(params)
    return {
        om: {p: range_map[p.file_index] for p in paths}
        for om, paths in path_map.items()
    }


class GateRun(NamedTuple):
    om: ma.OM
    boundaries: Path
    color_ranges: dict[ma.Color, int]
    fcs: ma.IndexedPath
    out: Path | None


def write_gate_inner(r: GateRun) -> tuple[ma.IndexedPath, Path | None]:
    gate_config = ga.read_gates(r.boundaries).sop1
    color = path_to_color(r.fcs.filepath)
    gs, _, _, _ = build_gating_strategy(
        gate_config,
        gate_config.scatter_gates[r.om].from_color(color),
        r.color_ranges,
        r.fcs.filepath,
        False,
    )
    if r.out is not None:
        with open(r.out, "wb") as f:
            fk.export_gatingml(gs, f)
    else:
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            fk.export_gatingml(gs, f)
    return (r.fcs, r.out)


def write_gate(
    om: ma.OM,
    boundaries: Path,
    params: Path,
    fcs: ma.IndexedPath,
    out: Path | None,
) -> tuple[ma.IndexedPath, Path | None]:
    range_map = ma.read_range_map(params)
    color_ranges = range_map[fcs.file_index]
    return write_gate_inner(GateRun(om, boundaries, color_ranges, fcs, out))


def write_all_gates(
    files: Path,
    boundaries: Path,
    params: Path,
    out_dir: Path | None,
    threads: int | None = None,
) -> list[tuple[ma.IndexedPath, Path | None]]:
    def make_out_path(om: ma.OM, p: Path) -> Path | None:
        color = from_maybe("rainbow", path_to_color(p))
        fn = f"{om}-{color}.xml"
        return fmap_maybe(lambda p: p / fn, out_dir)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    path_range_map = read_path_range_map(files, params)

    runs = [
        GateRun(
            om,
            boundaries,
            color_ranges,
            p,
            make_out_path(om, p.filepath),
        )
        for om, path_ranges in path_range_map.items()
        for p, color_ranges in path_ranges.items()
    ]

    if threads is None:
        return list(map(write_gate_inner, runs))
    else:
        with Pool(threads) as pl:
            return pl.map(write_gate_inner, runs)

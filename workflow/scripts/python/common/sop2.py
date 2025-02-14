import sys
import os
import numpy as np
import numpy.typing as npt
from pathlib import Path
import flowkit as fk  # type: ignore
from typing import NamedTuple
from multiprocessing import Pool
from common.io import read_fcs
import common.metadata as ma
import common.gating as ga
from common.functional import fmap_maybe, from_maybe, partition


def build_gating_strategy(
    gs: ga.SOP2Gates,
    rm: ma.RangeMap,
    fcs: ma.FcsCompensationMeta,
    scatteronly: bool,
) -> tuple[fk.GatingStrategy, fk.Sample, npt.NDArray[np.bool_]]:
    om = fcs.filemeta.machine.om
    material = fcs.filemeta.material
    fcs_path = fcs.indexed_path.filepath
    color = fcs.filemeta.color
    g_strat = fk.GatingStrategy()

    # Begin by adding the bead population scatter gates according to hardcoded
    # sample ranges

    scatter_bounds = gs.scatter_gates[om].from_material(material).from_color(color)
    bead_gate = ga.build_scatter_gate(scatter_bounds)
    g_strat.add_gate(bead_gate, ("root",))

    # The color gates are automatically placed according to events, so read
    # events, make a flowkit Sample, then gate out the beads
    parsed = read_fcs(fcs_path)
    smp = fk.Sample(parsed.events, sample_id=str(fcs_path.name))

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    if scatteronly:
        return g_strat, smp, mask

    # Apply logicle transform to each color channel. In this case there should
    # be relatively few events in the negative range, so A should be 0. Set M to
    # be 4.5 (sane default). I don't feel like getting the original range data
    # for each channel so just use the max of all of them (T = max). Then set W
    # according to 5% negative heuristic (see Parks et al (the Logicle Paper)
    # for formulas/rationale for doing this).
    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    trans = ga.transform_colors(
        gs.transform_config,
        df_beads,
        rm[fcs.indexed_path.file_index],
    )

    for c, t in trans.items():
        g_strat.add_transform(c.logicle_id, t)

    smp.apply_transform({c.value: t for c, t in trans.items()})
    df_beads_x = smp.as_dataframe(source="xform", event_mask=mask)

    x = df_beads_x[color.value].values
    r = ga.make_min_density_serial_gates(gs.autogate_config, x, 2)
    # gate_results = {color.value: r}

    for i, s in enumerate(r.xintervals):
        g_strat.add_gate(color.to_1d_gate(i, (s.x0, s.x1)), ("root", "beads"))

    return g_strat, smp, mask


def read_paths(files: Path) -> set[ma.FcsCompensationMeta]:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    fs = ma.read_files(files)
    return set(
        ma.FcsCompensationMeta(f.indexed_path, f.filemeta)
        for f in fs
        if f.filemeta.machine.om not in ma.NO_SCATTER
        and isinstance(f.filemeta, ma.CompensationMeta)
    )


def group_by_matrix(
    cs: set[ma.FcsCompensationMeta],
) -> dict[ma.Matrix, set[ma.FcsCompensationMeta]]:
    m1, rest1 = partition(lambda x: x.filemeta.matrix is ma.Matrix.Matrix1, cs)
    m4, rest4 = partition(lambda x: x.filemeta.matrix is None, rest1)
    m2, m3 = partition(lambda x: x.filemeta.matrix is ma.Matrix.Matrix2, rest4)
    return {
        ma.Matrix.Matrix1: set(m1),
        ma.Matrix.Matrix2: set(m2 + m4),
        ma.Matrix.Matrix3: set(m3 + m4),
    }


# def read_path_map(files: Path) -> FileMap:
#     """Read a tsv like "index, filepath" and return paths for SOP 1."""
#     calibrations = read_paths(files)
#     return {
#         om: set(x.indexed_path for x in gs)
#         for om, gs in groupby(calibrations, lambda c: c.filemeta.machine.om)
#     }


class CompensationRun(NamedTuple):
    fcs: ma.FcsCompensationMeta
    gate_config: ga.SOP2Gates
    color_ranges: ma.RangeMap
    out_dir: Path | None

    # TODO not DRY
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


def write_gate_inner(r: CompensationRun) -> tuple[ma.IndexedPath, Path | None]:
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
#     return write_gate_inner(CompensationRun(om, boundaries, color_ranges, fcs, out))


def write_all_gates(
    files: Path,
    boundaries: Path,
    params: Path,
    out_dir: Path | None,
    threads: int | None = None,
) -> list[tuple[ma.IndexedPath, Path | None]]:

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    comps = read_paths(files)

    gs = ga.read_gates(boundaries).sop2
    range_map = ma.read_range_map(params)

    runs = [CompensationRun(c, gs, range_map, out_dir) for c in comps]

    if threads is None:
        return list(map(write_gate_inner, runs))
    else:
        with Pool(threads) as pl:
            return pl.map(write_gate_inner, runs)

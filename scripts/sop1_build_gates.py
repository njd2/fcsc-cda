#! /usr/bin/env python3

import math
from tempfile import NamedTemporaryFile
import argparse
import sys
import os
from pathlib import Path
import flowkit as fk  # type: ignore
import fcsparser as fp  # type: ignore
from typing import Any, NamedTuple, IO, NewType
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde  # type: ignore
from bokeh.plotting import show, output_file
from bokeh.layouts import row, column
from bokeh.models import TabPanel, Tabs

PeakCoord = tuple[float, float]
PeakCoords = list[PeakCoord]

OM = NewType("OM", str)
Color = NewType("Color", str)

RAINBOW_PAT = "RCP-30"

NO_SCATTER = [
    OM("LMNXSEA_CellStream-1"),
    OM("LMNXSEA_CellStream-2"),
    OM("LMNXSEA_ImageStreamX-1"),
]

COLORS = [*map(Color, ["v450", "v500", "fitc", "pc55", "pe", "pc7", "apc", "ac7"])]

FILE_ORDER = [
    f"FC-{x}_SOP"
    for x in [
        "V450",
        "V500-C",
        "FITC",
        "PerCP-Cy5.5",
        "PE",
        "PE-Cy7",
        "APC",
        "APC-Cy7",
    ]
]

COLOR_MAP = dict(zip(FILE_ORDER, COLORS))

LogicleM = 4.5


class GateRanges(NamedTuple):
    fsc_min: int
    fsc_max: int
    ssc_min: int
    ssc_max: int


def path_to_color(p: Path) -> Color | None:
    return next((v for k, v in COLOR_MAP.items() if k in p.name), None)


def build_gating_strategy(gs: GateRanges) -> Any:
    dim_fsc = fk.Dimension("fsc_a", range_min=gs.fsc_min, range_max=gs.fsc_max)
    dim_ssc = fk.Dimension("ssc_a", range_min=gs.ssc_min, range_max=gs.ssc_max)

    rect_top_left_gate = fk.gates.RectangleGate(
        "beads",
        dimensions=[dim_fsc, dim_ssc],
    )

    g_strat = fk.GatingStrategy()
    g_strat.add_gate(rect_top_left_gate, gate_path=("root",))

    return g_strat


def find_differential_peaks(
    positions: npt.NDArray[np.float32],
    pdf: npt.NDArray[np.float32],
    dd_pdf: npt.NDArray[np.float32],
    signal: int,
) -> PeakCoords:
    mask = np.zeros((positions.size), dtype=bool)
    mask[1:-1] = dd_pdf == signal
    x = positions[mask].tolist()
    y = pdf[mask].tolist()
    coords = list(zip(x, y))
    coords.sort(key=lambda x: -x[1])
    return coords


def find_density_peaks(
    x: npt.NDArray[np.float32],
    n: int = 512,
    bw_method: str | float = "scott",
) -> tuple[PeakCoords, PeakCoords]:
    """Find density peaks in vector x using "double diff" method.

    Density is computed with n evenly spaced intervals over x using a gaussian
    kernel.

    Specifically, find the places where the 1st derivative goes immediately from
    positive to negative (peaks) or the reverse (valleys).

    Return list of peaks and valley positions.
    """

    if x.size < 3:
        raise ValueError("need at least two points to compute peaks/valleys")
    positions = np.linspace(x.min(), x.max(), n)
    kernel = gaussian_kde(x, bw_method=bw_method)
    pdf = kernel(positions)
    # TODO this will fail if the peak consists of more than one point (ie
    # multiple consecutive points are tied for the local maximum); in practice
    # these will be rare, but don't say I didn't warn you when this turns out
    # not to be forever ;)
    dd_pdf = np.diff(np.sign(np.diff(pdf)))
    peaks = find_differential_peaks(positions, pdf, dd_pdf, -2)
    valleys = find_differential_peaks(positions, pdf, dd_pdf, 2)
    return (peaks, valleys)


def make_min_density_serial_gates(
    x: npt.NDArray[np.float32],
    k: int,
    bw: float,
) -> list[float]:
    """Gate vector x by minimum density between at most k peaks."""
    peaks, valleys = find_density_peaks(x, bw_method=bw)
    if len(peaks) < 2:
        raise ValueError(f"could not make serial gates, got {len(peaks)} peaks")
    top_peaks = sorted(peaks[: min(k, len(peaks))], key=lambda p: p[0])
    peak_intervals = [(p0[0], p1[0]) for p0, p1 in zip(top_peaks[:-1], top_peaks[1:])]
    contained_valleys = [
        [v for v in valleys if x0 < v[0] < x1] for x0, x1 in peak_intervals
    ]
    if any([len(v) == 0 for v in contained_valleys]):
        raise ValueError("not all peak intervals contain a valley")
    min_valleys = [v[-1][0] for v in contained_valleys]
    return min_valleys


def apply_gates_to_sample(
    fcs_path: Path,
    gs: GateRanges,
    colors: list[Color],
    bead_color: Color | None,
) -> Any:
    g_strat = build_gating_strategy(gs)
    _, df = fp.parse(fcs_path, channel_naming="$PnN")
    smp = fk.Sample(df, sample_id=str(fcs_path.name))

    fsc_max = df["fsc_a"].max()
    ssc_max = df["ssc_a"].max()

    res = g_strat.gate_sample(smp)
    mask = res.get_gate_membership("beads")

    p0 = smp.plot_scatter(
        "fsc_a",
        "ssc_a",
        source="raw",
        highlight_mask=mask,
        x_max=min(gs.fsc_max * 5, fsc_max),
        y_max=min(gs.ssc_max * 5, ssc_max),
    )

    # p1 = smp.plot_scatter(
    #     "fsc_h",
    #     "ssc_h",
    #     source="raw",
    #     highlight_mask=mask,
    # )

    df_beads = smp.as_dataframe(source="raw", event_mask=mask)
    smp_beads = fk.Sample(df_beads, sample_id=str(fcs_path.name))
    color_max = max([df_beads[c].max() for c in COLORS])
    trans = {}
    # see Parks et al (the Logicle Paper) for formulas/rationale for doing this
    for c in COLORS:
        arr = df_beads[c].values
        arr_neg = arr[arr < 0]
        if arr_neg.size < 10:
            trans[c] = fk.transforms.LogicleTransform(color_max, 1.0, LogicleM, 0)
        else:
            low_ref = np.quantile(arr_neg, 0.05)
            best_W = (LogicleM - math.log10(color_max / abs(low_ref))) / 2
            trans[c] = fk.transforms.LogicleTransform(color_max, best_W, LogicleM, 0)
    smp_beads.apply_transform(trans)
    df_beads_x = smp_beads.as_dataframe(source="xform")
    ps = []
    for c in colors:
        p = smp_beads.plot_histogram(c, source="xform", x_range=(-0.2, 1))
        if bead_color is not None and c == bead_color:
            gate = make_min_density_serial_gates(df_beads_x[c].values, 2, 0.2)
            p.vspan(x=[gate], color="red")
        elif bead_color is None:
            gates = make_min_density_serial_gates(df_beads_x[c].values, 8, 0.05)
            for g in gates:
                p.vspan(x=[g], color="red")
        ps.append(p)
    return row(p0, *ps)


def read_path_map(files_path: Path) -> dict[OM, list[Path]]:
    """Read a tsv like "index, filepath" and return paths for SOP 1."""
    df = pd.read_table(files_path, usecols=[1], names=["file_path"])
    acc: dict[OM, list[Path]] = {}
    for _, x in df["file_path"].items():
        p = Path(x)
        xs = p.name.split("_")
        om = OM(f"{xs[2]}_{xs[3]}")
        if xs[5] == "SOP-01" and om not in NO_SCATTER:
            if om not in acc:
                acc[om] = []
            acc[om] += [p]
    for k in acc:
        acc[k].sort(
            key=lambda x: next((i for i, o in enumerate(FILE_ORDER) if o in x.name), -1)
        )
    return acc


# def write_blank_gate_ranges(files_path: Path) -> None:
#     path_map = read_path_map(files_path)
#     no_rainbow = [(om, False, 0, 1e12, 0, 1e12) for om in path_map]
#     rainbow = [(om, True, 0, 1e12, 0, 1e12) for om in path_map]
#     df = pd.DataFrame(
#         no_rainbow + rainbow,
#         columns=[
#             "om",
#             "is_rainbow",
#             "fsc_min",
#             "fsc_max",
#             "ssc_min",
#             "ssc_max",
#         ],
#     )
#     df.to_csv(sys.stdout, index=False, sep="\t")


def read_gate_ranges(ranges_path: Path) -> dict[tuple[OM, bool], GateRanges]:
    df = pd.read_table(ranges_path)
    return {
        (om, is_rainbow): GateRanges(int(f0), int(f1), int(s0), int(s1))
        for om, is_rainbow, f0, f1, s0, s1 in df.itertuples(index=False)
    }


def make_plots(
    ranges_path: Path,
    files_path: Path,
    om: OM,
    rainbow: bool,
    colors: list[Color],
) -> None:
    all_gs = read_gate_ranges(ranges_path)
    path_map = read_path_map(files_path)
    paths = path_map[om]

    non_rainbow_tab = TabPanel(
        child=column(
            *[
                apply_gates_to_sample(
                    p,
                    all_gs[(om, False)],
                    colors,
                    path_to_color(p),
                )
                for p in paths
                if RAINBOW_PAT not in p.name
            ]
        ),
        title="Non Rainbow",
    )
    rainbow_plot = next(
        (
            apply_gates_to_sample(p, all_gs[(om, True)], colors, None)
            for p in paths
            if RAINBOW_PAT in p.name
        ),
        None,
    )
    if rainbow_plot:
        rainbow_tab = TabPanel(child=column(rainbow_plot), title="Rainbow")
        page = Tabs(
            tabs=(
                [rainbow_tab, non_rainbow_tab]
                if rainbow
                else [non_rainbow_tab, rainbow_tab]
            )
        )
    else:
        page = Tabs(tabs=[non_rainbow_tab])
    # open but don't close temp file to save plot
    tf = NamedTemporaryFile(suffix="_sop1.html", delete_on_close=False, delete=False)
    output_file(tf.name)
    show(page)


def write_gate(
    ranges_path: Path,
    om: OM,
    gate_handle: IO[bytes],
    rainbow: bool,
) -> None:
    all_gs = read_gate_ranges(ranges_path)
    gs = build_gating_strategy(all_gs[(om, rainbow)])
    fk.export_gatingml(gs, gate_handle)


def write_all_gates(
    files_path: Path,
    ranges_path: Path,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_map = read_path_map(files_path)
    for om in path_map:

        def go(rainbow: bool) -> None:
            r = "rcp" if rainbow else "fc"
            with open(out_dir / f"{prefix}_{str(om)}_{r}.xml", "wb") as ho:
                write_gate(ranges_path, om, ho, rainbow)

        go(True)
        go(False)


def list_oms(files_path: Path) -> None:
    xs = read_path_map(files_path)
    for x in xs:
        print(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    list_parser = subparsers.add_parser("list", help="list all org/machine IDs")
    list_parser.add_argument("files", help="path to list of files")

    plot_parser = subparsers.add_parser("plot", help="plot gates for org/machine ID")
    plot_parser.add_argument("om", help="org/machine ID")
    plot_parser.add_argument("files", help="path to list of files")
    plot_parser.add_argument("gates", help="path to list of gate ranges")
    plot_parser.add_argument(
        "-R",
        "--rainbow",
        action="store_true",
        help="show rainbow first (for extreme laziness)",
    )
    plot_parser.add_argument(
        "-c",
        "--colors",
        help="comma separated list of colors to include",
    )

    write_gate_parser = subparsers.add_parser(
        "write_gate",
        help="write gate for org/machine ID",
    )
    write_gate_parser.add_argument("om", help="org/machine ID")
    write_gate_parser.add_argument("gates", help="path to list of gate ranges")
    write_gate_parser.add_argument(
        "-R",
        "--rainbow",
        action="store_true",
        help="save rainbow gate",
    )

    write_gates_parser = subparsers.add_parser("write_gates", help="write all gates")
    write_gates_parser.add_argument("files", help="path to list of files")
    write_gates_parser.add_argument("gates", help="path to list of gate ranges")
    write_gates_parser.add_argument("outdir", help="output directory")
    write_gates_parser.add_argument("prefix", help="output file prefix ")

    parsed = parser.parse_args(sys.argv[1:])

    if parsed.cmd == "list":
        list_oms(Path(parsed.files))

    if parsed.cmd == "plot":
        make_plots(
            Path(parsed.gates),
            Path(parsed.files),
            OM(parsed.om),
            parsed.rainbow,
            COLORS if parsed.colors is None else parsed.colors.split(","),
        )

    if parsed.cmd == "write_gate":
        # do some POSIX gymnastics to get stdout to accept a bytestream
        with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as f:
            write_gate(Path(parsed.gates), OM(parsed.om), f, parsed.rainbow)

    if parsed.cmd == "write_gates":
        write_all_gates(
            Path(parsed.files),
            Path(parsed.gates),
            Path(parsed.outdir),
            parsed.prefix,
        )


main()

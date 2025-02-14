import csv
from pathlib import Path
from typing import Any
import common.sop1 as s1


def main(smk: Any) -> None:
    files_path = Path(smk.input["files"])
    gates_path = Path(smk.input["gates"])
    params_path = Path(smk.input["params"])

    out_path = Path(smk.output[0])

    out_dir = out_path.parent

    results = s1.write_all_gates(
        files_path,
        gates_path,
        params_path,
        out_dir,
        smk.threads,
    )

    with open(out_path, "wt") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file_index", "fcs_path", "gate_path"])
        for fcs, gate_out in results:
            w.writerow([str(fcs.file_index), str(fcs.filepath), str(gate_out)])


main(snakemake)  # type: ignore

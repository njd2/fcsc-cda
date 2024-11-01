#! /usr/bin/env python3

import re
import sys
import argparse
from pathlib import Path
import multiprocessing as plaid  # when ludicrous speed just isn't enough
import getpass
import requests  # type: ignore
from requests.auth import HTTPBasicAuth  # type: ignore
from dataclasses import dataclass
from typing import NamedTuple

URL = "https://labcas.jpl.nasa.gov/nist/data-access-api/download"


class Target(NamedTuple):
    path: Path
    params: dict[str, str]
    cols: list[str]


def to_target(cols: list[str]) -> Target:
    #target_path = Path(cols[0]) / cols[5] / cols[6] / cols[8] / cols[13]
    target_path = Path(cols[13])

    # (0) Working group (always "WG2") /
    # (1) Instrument Code /
    # (2) Site Code /
    # (3) ProtocolID /
    # (4) Experiment Type /
    # (5) Sample Name /
    # (6) PrincipleContactID /
    # (7) DataProcessingLevel (always "Raw") /
    # (8) StudyID (always "FCSC_WG2-001") /
    # (9) Material Code /
    # (10) ExperimentID /
    # (11) Replicate /
    # (12) DataFormat (always FCS) /
    # (13) File Name
    indices = [0, 6, 5, 9, 1, 8, 2, 3, 4, 7, 10, 11, 12, 13]
    params = {
        "id": "/".join(
            [
                "NIST_Flow_Cytometry_Standards_Consortium",
                *[cols[i] for i in indices],
            ]
        )
    }
    return Target(target_path, params, cols)


def read_targets(path: Path) -> list[Target]:
    with open(path, "r") as i:
        # skip first row
        next(i, None)
        return [to_target([str(c) for c in x.strip().split("\t")]) for x in i]


@dataclass(frozen=True)
class Downloader:
    username: str
    password: str
    dryrun: bool
    directory: Path
    x_pattern: str | None

    def __call__(self, t: Target) -> str:

        if self.x_pattern is not None and re.match(self.x_pattern, str(t.path)):
            return f"Excluding {t.path}"

        fcs_path = self.directory / t.path

        if fcs_path.exists():
            return f"Skipping {fcs_path}"

        if self.dryrun:
            return f"Will download {fcs_path}"

        fcs_path.parent.mkdir(exist_ok=True, parents=True)

        resp = requests.get(
            "https://labcas.jpl.nasa.gov/nist/data-access-api/auth",
            auth=HTTPBasicAuth(self.username, self.password),
        )
        token = resp.text

        try:
            resp = requests.get(
                URL,
                params=t.params,
                headers={"Authorization": "Bearer {}".format(token)},
                stream=True,
            )
            if resp.status_code == 200:
                with open(fcs_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return f"Downloaded: {fcs_path}"
            else:
                return (
                    f"Error {resp.status_code} "
                    f"- {t.cols[0]}, {t.cols[5]}, {t.cols[6]}, {t.cols[13]}"
                )
        # TODO be more specific
        except Exception as e:
            return str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="download lots of FCS files")

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-d", "--directory", default="downloaded")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-u", "--username", default=None)
    parser.add_argument("-n", "--dryrun", action="store_true")
    parser.add_argument("-r", "--firstrows", type=int, default=None)
    parser.add_argument("-x", "--exclude", default=None)

    args = parser.parse_args()

    username = input("Username: ") if args.username is None else args.username
    password = getpass.getpass(f"Password for {username} at nist: ")
    rows = args.firstrows

    dl = Downloader(username, password, args.dryrun, args.directory, args.exclude)

    with plaid.Pool(args.workers) as p:
        targets = read_targets(args.input)
        import time

        t0 = time.time()
        results = p.map(dl, targets[:rows] if rows is not None else targets)
        print(time.time() - t0)

    with sys.stdout if args.output is None else open(args.output, "w") as f:
        for r in results:
            f.write(r + "\n")


main()

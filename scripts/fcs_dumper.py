#! /usr/bin/env python3

import argparse
import sys
from common.io import read_fcs, TabularTEXT30, ParsedEvents


def print_version(fcs: ParsedEvents) -> dict[str, str]:
    version = 3.0 if isinstance(fcs.meta.standard, TabularTEXT30) else 3.1
    return {"FCS version": str(version)}


def print_header(fcs: ParsedEvents) -> dict[str, str]:
    return {k: str(v) for k, v in fcs.header._asdict().items()}


def print_params(fcs: ParsedEvents) -> dict[str, str]:
    return {f"$P{p.index_}{p.ptype.value}": str(p.value) for p in fcs.meta.params}


def print_standard(fcs: ParsedEvents, remove_none: bool) -> dict[str, str]:
    return {
        k: str(v)
        for k, v in fcs.meta.standard.dict().items()
        if v is not None or not remove_none
    }


def print_nonstandard(fcs: ParsedEvents) -> dict[str, str]:
    return {k: v for k, v in fcs.meta.nonstandard.items()}


def print_deviant(fcs: ParsedEvents) -> dict[str, str]:
    return {str(k): v for k, v in fcs.meta.deviant.items()}


def print_events(fcs: ParsedEvents) -> None:
    print(fcs.events.describe())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to FCS file to parse")
    parser.add_argument("-V", "--version", help="print version", action="store_true")
    parser.add_argument("-H", "--header", help="print header", action="store_true")
    parser.add_argument("-p", "--params", help="print parameters", action="store_true")
    parser.add_argument(
        "-s",
        "--standard",
        help="print standard keywords",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--nonstandard",
        help="print non-standard keywords",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--deviant",
        help="print deviant keywords",
        action="store_true",
    )
    parser.add_argument(
        "-r", "--remove", help="remove None values", action="store_true"
    )
    parser.add_argument(
        "-e",
        "--events",
        help="print event statistics",
        action="store_true",
    )
    parsed = parser.parse_args(sys.argv[1:])

    fcs = read_fcs(parsed.path)

    acc: dict[str, str] = {}

    if parsed.version:
        acc.update(print_version(fcs))

    if parsed.header:
        acc.update(print_header(fcs))

    if parsed.params:
        acc.update(print_params(fcs))

    if parsed.standard:
        acc.update(print_standard(fcs, parsed.remove))

    if parsed.nonstandard:
        acc.update(print_nonstandard(fcs))

    if parsed.deviant:
        acc.update(print_deviant(fcs))

    for k, v in acc.items():
        print(f"{k}: {v}")

    if parsed.events:
        print_events(fcs)


main()

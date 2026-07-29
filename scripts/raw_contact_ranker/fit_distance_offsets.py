#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  # inserts src/ into sys.path

from raw_contact_ranker.common import load_config
from raw_contact_ranker.offsets import fit_distance_offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    fit_distance_offsets(load_config(args.config))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.features import extract_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    report = extract_features(load_config(args.config), args.pairs.resolve())
    print(f"Extracted {report['pair_count']:,} exact pair features")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evaluation import evaluate_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_baselines(
        load_config(args.config), args.manifest.resolve(), args.output.resolve()
    )
    print(f"Wrote baseline contract with {len(report['warnings'])} warnings")


if __name__ == "__main__":
    main()

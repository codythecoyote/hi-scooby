#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evidence import validate_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_evidence(
        load_config(args.config), args.reference.resolve(), args.output.resolve()
    )
    print(f"Count conservation: {report['count_conservation']}; warnings: {len(report['warnings'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concordance-gate", type=Path, required=True)
    args = parser.parse_args()
    with args.concordance_gate.open() as handle:
        report = json.load(handle)
    print(
        sum(
            row.get("passed") is True
            for row in report.get("outputs", {}).values()
        )
    )


if __name__ == "__main__":
    main()

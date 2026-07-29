#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=("selected_feature_set", "selected_epoch_count"),
        required=True,
    )
    args = parser.parse_args()
    with args.selection.open() as handle:
        report = json.load(handle)
    value = report[args.field]
    if value is None:
        raise SystemExit(f"Selection field {args.field} is null")
    print(value)


if __name__ == "__main__":
    main()

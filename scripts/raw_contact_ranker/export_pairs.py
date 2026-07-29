#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.pairs import export_canonical_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = args.output_root or Path(config["outputs"]["data_root"])
    report = export_canonical_pairs(config, output_root.resolve())
    print(f"Wrote {report['canonical_pairs']:,} canonical pairs to {report['output']}")


if __name__ == "__main__":
    main()

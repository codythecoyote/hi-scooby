#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evidence import export_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    config = load_config(args.config)
    report = export_evidence(
        config, args.pairs.resolve(), splits=args.splits, seed=args.seed
    )
    print(f"Wrote evidence for {len(report['contexts'])} contexts")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.selection import select_shared_feature_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select_shared_feature_set(
        load_config(args.config),
        args.metrics,
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["topology_training_authorized"]:
        raise SystemExit(
            "Exposure-only was selected; shared topology training is blocked"
        )


if __name__ == "__main__":
    main()

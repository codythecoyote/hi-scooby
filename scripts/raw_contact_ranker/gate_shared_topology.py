#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.topology import evaluate_shared_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--freeze-test", action="store_true")
    parser.add_argument("--frozen-release", type=Path)
    parser.add_argument("--test-lock", type=Path)
    args = parser.parse_args()
    print(
        f"[rate10k] Applying shared-topology gates on {args.split} "
        f"predictions: {args.predictions.resolve()}",
        flush=True,
    )
    report = evaluate_shared_topology(
        load_config(args.config),
        prediction_path=args.predictions.resolve(),
        output=args.output.resolve(),
        split=args.split,
        freeze_test=args.freeze_test,
        frozen_release=(
            args.frozen_release.resolve() if args.frozen_release else None
        ),
        test_lock=args.test_lock.resolve() if args.test_lock else None,
    )
    passed = sum(report["checks"].values())
    print(
        f"[rate10k] Shared topology promoted: {report['promoted']} "
        f"({passed}/{len(report['checks'])} checks passed); "
        f"report={args.output.resolve()}",
        flush=True,
    )
    if not report["promoted"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

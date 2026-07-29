#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.calibration import evaluate_frozen_calibration
from raw_contact_ranker.common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        "[rate10k] Evaluating the frozen NB2 calibration once on the "
        "untouched test split",
        flush=True,
    )
    report = evaluate_frozen_calibration(
        load_config(args.config),
        predictions=args.predictions.resolve(),
        calibration_path=args.calibration.resolve(),
        rollout_path=args.rollout.resolve(),
        frozen_release=args.frozen_release.resolve(),
        test_lock=args.test_lock.resolve(),
        output=args.output.resolve(),
    )
    passed = sum(report["checks"].values())
    print(
        f"[rate10k] Frozen calibration accepted: {report['accepted']} "
        f"({passed}/{len(report['checks'])} checks passed); "
        f"report={args.output.resolve()}",
        flush=True,
    )
    if not report["accepted"]:
        raise SystemExit("Frozen test calibration gate failed")


if __name__ == "__main__":
    main()

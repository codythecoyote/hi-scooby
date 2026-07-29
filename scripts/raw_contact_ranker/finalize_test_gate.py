#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config  # noqa: F401
from raw_contact_ranker.release import finalize_test_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    parser.add_argument("--exact-evaluation", type=Path, required=True)
    parser.add_argument("--topology-gate", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path, required=True)
    parser.add_argument("--context-test-gate", type=Path)
    parser.add_argument("--context-prediction-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_test_gate(
        frozen_release=args.frozen_release.resolve(),
        test_lock=args.test_lock.resolve(),
        exact_evaluation=args.exact_evaluation.resolve(),
        topology_gate=args.topology_gate.resolve(),
        calibration_gate=args.calibration_gate.resolve(),
        context_test_gate=(
            args.context_test_gate.resolve() if args.context_test_gate else None
        ),
        context_prediction_report=(
            args.context_prediction_report.resolve()
            if args.context_prediction_report
            else None
        ),
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["accepted"]:
        raise SystemExit("Final untouched-test gate failed")


if __name__ == "__main__":
    main()

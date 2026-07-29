#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.release import freeze_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-evaluation", type=Path, required=True)
    parser.add_argument("--power-gate", type=Path, required=True)
    parser.add_argument("--topology-gate", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--context-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = freeze_release(
        load_config(args.config),
        checkpoint=args.checkpoint.resolve(),
        validation_evaluation=args.validation_evaluation.resolve(),
        power_gate=args.power_gate.resolve(),
        topology_gate=args.topology_gate.resolve(),
        calibration_gate=args.calibration_gate.resolve(),
        selection=args.selection.resolve(),
        rollout=args.rollout.resolve(),
        context_gate=(
            args.context_gate.resolve() if args.context_gate else None
        ),
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

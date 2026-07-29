#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context import build_context_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--power-gate", type=Path, required=True)
    parser.add_argument("--concordance-gate", type=Path, required=True)
    parser.add_argument("--calibration-gate", type=Path, required=True)
    parser.add_argument("--extension-gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_context_rollout(
        load_config(args.config),
        power_gate=args.power_gate.resolve(),
        concordance_gate=args.concordance_gate.resolve(),
        calibration_gate=args.calibration_gate.resolve(),
        extension_gate=(
            args.extension_gate.resolve() if args.extension_gate else None
        ),
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import REPO_ROOT
from raw_contact_ranker.common import atomic_json
from raw_contact_ranker.promotion import assess_epoch3_promotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stop a full-model run after epoch 3 unless local recovery is visible."
    )
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.training.open() as handle:
        training = json.load(handle)
    with args.metrics.open() as handle:
        metrics = json.load(handle)
    with args.baselines.open() as handle:
        baselines = json.load(handle)
    report = assess_epoch3_promotion(training, metrics, baselines)
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        print(
            "Full model failed the epoch-3 promotion gate; continuation is blocked.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()

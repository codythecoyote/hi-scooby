#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--freeze-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_checkpoint(
        load_config(args.config),
        args.checkpoint.resolve(),
        args.split,
        output=args.output.resolve(),
        freeze_test=args.freeze_test,
    )
    print(f"Evaluated {report['candidate_pairs']:,} {args.split} pairs")


if __name__ == "__main__":
    main()

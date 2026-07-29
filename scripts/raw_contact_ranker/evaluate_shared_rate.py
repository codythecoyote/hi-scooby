#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.exact_rate import evaluate_exact_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--freeze-test", action="store_true")
    parser.add_argument("--frozen-release", type=Path)
    parser.add_argument("--test-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"[rate10k] Evaluating frozen shared topology on {args.split}: "
        f"checkpoint={args.checkpoint.resolve()}",
        flush=True,
    )
    report = evaluate_exact_checkpoint(
        load_config(args.config),
        args.checkpoint.resolve(),
        split=args.split,
        output=args.output.resolve(),
        freeze_test=args.freeze_test,
        frozen_release=(
            args.frozen_release.resolve() if args.frozen_release else None
        ),
        test_lock=args.test_lock.resolve() if args.test_lock else None,
    )
    print(
        f"[rate10k] Shared {args.split} evaluation complete: "
        f"gain_per_event={report['gain_per_event']}, "
        f"events={report['events']:.0f}, report={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context_head import predict_context_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-test", action="store_true")
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"[rate10k] Generating authorized one-shot context predictions: "
        f"{args.checkpoint.resolve()}",
        flush=True,
    )
    report = predict_context_test(
        load_config(args.config),
        checkpoint_path=args.checkpoint.resolve(),
        output=args.output.resolve(),
        freeze_test=args.freeze_test,
        frozen_release=args.frozen_release.resolve(),
        test_lock=args.test_lock.resolve(),
    )
    print(
        f"[rate10k] Context test predictions written: "
        f"{report['prediction_path']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

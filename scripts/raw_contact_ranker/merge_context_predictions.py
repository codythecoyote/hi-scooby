#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context_head import merge_context_release_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--context-gate", type=Path, required=True)
    parser.add_argument("--onehot-test", type=Path, required=True)
    parser.add_argument("--rna-test", type=Path, required=True)
    parser.add_argument("--permuted-rna-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    args = parser.parse_args()
    report = merge_context_release_predictions(
        load_config(args.config),
        context_gate_path=args.context_gate.resolve(),
        test_predictions={
            "onehot": args.onehot_test.resolve(),
            "rna": args.rna_test.resolve(),
            "rna_permuted": args.permuted_rna_test.resolve(),
        },
        frozen_release=args.frozen_release.resolve(),
        test_lock=args.test_lock.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

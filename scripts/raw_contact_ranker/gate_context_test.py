#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context_gate import evaluate_context_test_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--validation-context-gate", type=Path, required=True)
    parser.add_argument("--shared-test-predictions", type=Path, required=True)
    parser.add_argument("--onehot-test-predictions", type=Path, required=True)
    parser.add_argument("--rna-test-predictions", type=Path, required=True)
    parser.add_argument(
        "--permuted-rna-test-predictions", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    args = parser.parse_args()
    print(
        "[rate10k] Applying frozen context-extension comparisons to the "
        "untouched test split",
        flush=True,
    )
    report = evaluate_context_test_extension(
        load_config(args.config),
        validation_context_gate=args.validation_context_gate.resolve(),
        shared_test_predictions=args.shared_test_predictions.resolve(),
        onehot_test_predictions=args.onehot_test_predictions.resolve(),
        rna_test_predictions=args.rna_test_predictions.resolve(),
        permuted_rna_test_predictions=(
            args.permuted_rna_test_predictions.resolve()
        ),
        frozen_release=args.frozen_release.resolve(),
        test_lock=args.test_lock.resolve(),
        output=args.output.resolve(),
    )
    passed = sum(report["checks"].values())
    print(
        f"[rate10k] Final context test gate accepted: {report['accepted']} "
        f"({passed}/{len(report['checks'])} outputs passed); "
        f"report={args.output.resolve()}",
        flush=True,
    )
    if not report["accepted"]:
        raise SystemExit("Final context test gate failed")


if __name__ == "__main__":
    main()

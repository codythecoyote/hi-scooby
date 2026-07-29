#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context_gate import evaluate_context_extension


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--shared-predictions", type=Path, required=True)
    parser.add_argument("--onehot-report", type=Path, required=True)
    parser.add_argument("--rna-report", type=Path, required=True)
    parser.add_argument("--permuted-rna-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        "[rate10k] Evaluating one-hot, RNA, and RNA-permuted context "
        "topology heads with FDR control",
        flush=True,
    )
    report = evaluate_context_extension(
        load_config(args.config),
        shared_predictions=args.shared_predictions.resolve(),
        onehot_report=args.onehot_report.resolve(),
        rna_report=args.rna_report.resolve(),
        permuted_rna_report=args.permuted_rna_report.resolve(),
        output=args.output.resolve(),
    )
    accepted = sum(
        row["topology_accepted"] for row in report["outputs"].values()
    )
    print(
        f"[rate10k] Context topology gate complete: "
        f"{accepted}/{len(report['outputs'])} outputs promoted; "
        f"report={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

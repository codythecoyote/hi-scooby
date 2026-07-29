#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.context import evaluate_context_concordance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--power-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        "[rate10k] Comparing same-output pseudoreplicates with "
        "depth-matched cross-context controls",
        flush=True,
    )
    report = evaluate_context_concordance(
        load_config(args.config),
        power_gate=args.power_gate.resolve(),
        output=args.output.resolve(),
    )
    passed = sum(row["passed"] for row in report["outputs"].values())
    print(
        f"[rate10k] Context concordance complete: "
        f"{passed}/{len(report['outputs'])} outputs eligible; "
        f"report={args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

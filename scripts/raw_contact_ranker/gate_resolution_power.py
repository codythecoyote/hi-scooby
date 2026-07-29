#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.power import summarize_power_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--power-groups", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = Path(config["outputs"]["data_root"])
    output = (
        args.output
        or Path(config["outputs"]["results_root"]) / "power_gate.json"
    )
    power_groups = args.power_groups or data_root / "power_groups.parquet"
    print(
        f"[rate10k] Evaluating the fail-closed power gate from "
        f"{power_groups.resolve()}",
        flush=True,
    )
    report = summarize_power_audit(
        config,
        power_groups,
        output=output,
    )
    passed = sum(report["checks"].values())
    print(
        f"[rate10k] Resolution power eligible: {report['eligible']} "
        f"({passed}/{len(report['checks'])} checks passed); "
        f"report={output.resolve()}",
        flush=True,
    )
    if not report["eligible"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

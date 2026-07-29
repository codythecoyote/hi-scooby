#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evaluation import evaluate_rna_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--sequence-checkpoint", type=Path, required=True)
    parser.add_argument("--rna-checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or Path(config["outputs"]["results_root"]) / "rna" / "validation_gate.json"
    report = evaluate_rna_gate(
        config,
        args.sequence_checkpoint.resolve(),
        args.rna_checkpoint.resolve(),
        args.split,
        output.resolve(),
    )
    print(f"RNA supported: {report['rna_supported']}")


if __name__ == "__main__":
    main()

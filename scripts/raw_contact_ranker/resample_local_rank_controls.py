#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_contact_ranker.common import load_config
from raw_contact_ranker.sampling import resample_local_rank_controls


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace legacy chromosome-wide rank controls with controls from "
            "the same tile and distance band, without resampling likelihood controls."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/raw_contact_ranker.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    report = resample_local_rank_controls(load_config(args.config), seed=args.seed)
    print(report)


if __name__ == "__main__":
    main()

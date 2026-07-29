#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.evaluation import build_evaluation_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    args = parser.parse_args()
    report = build_evaluation_manifest(load_config(args.config))
    print(f"Froze {report['candidate_pairs']:,} validation candidate pairs")


if __name__ == "__main__":
    main()

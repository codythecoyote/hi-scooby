#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import default_config  # noqa: F401
from raw_contact_ranker.release import acquire_test_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    report = acquire_test_lock(
        args.frozen_release.resolve(), args.lock.resolve()
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

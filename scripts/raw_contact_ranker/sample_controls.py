#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.sampling import sample_controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--controls-per-event", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    report = sample_controls(
        load_config(args.config),
        controls_per_event=args.controls_per_event,
        seed=args.seed,
    )
    print(
        f"Wrote {report['event_controls']:,} event controls and "
        f"{report['rank_controls']:,} rank controls"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.resolution import prepare_resolution_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    args = parser.parse_args()
    print(
        f"[rate10k] Preparing aligned tiles and count-only Cooler from "
        f"{args.config.resolve()}",
        flush=True,
    )
    report = prepare_resolution_inputs(load_config(args.config))
    print(
        f"[rate10k] Resolution inputs ready: "
        f"tiles={report['tiles']['rows']:,}, "
        f"changed_starts={report['tiles']['changed_target_starts']:,}, "
        f"conserved_contacts={report['cooler']['target_count']:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

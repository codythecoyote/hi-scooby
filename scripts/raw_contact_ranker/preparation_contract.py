#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.provenance import (
    create_preparation_contract,
    migrate_static_annotation_contract,
    verify_preparation_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--migrate-static-annotations-from")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.verify and args.migrate_static_annotations_from:
        parser.error("--verify and --migrate-static-annotations-from are exclusive")
    if args.verify:
        verify_preparation_contract(config, args.output.resolve())
        print("Preparation provenance matches the current clean checkout and sources")
    elif args.migrate_static_annotations_from:
        migrated = migrate_static_annotation_contract(
            config,
            args.output.resolve(),
            expected_prior_commit=args.migrate_static_annotations_from,
        )
        print(
            "Migrated preparation contract for static annotations: "
            f"{migrated['git_commit']}"
        )
    else:
        create_preparation_contract(config, args.output.resolve())
        print(f"Wrote preparation contract: {args.output.resolve()}")


if __name__ == "__main__":
    main()

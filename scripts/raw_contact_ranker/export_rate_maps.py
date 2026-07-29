#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import default_config
from raw_contact_ranker.common import load_config
from raw_contact_ranker.export import export_rate_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config())
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--frozen-release", type=Path, required=True)
    parser.add_argument("--final-test-gate", type=Path, required=True)
    parser.add_argument("--canonical-output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"[rate10k] Exporting immutable canonical rates and 10 kb map views "
        f"for {len(args.predictions)} frozen prediction splits",
        flush=True,
    )
    report = export_rate_maps(
        load_config(args.config),
        prediction_paths=[path.resolve() for path in args.predictions],
        calibration_path=args.calibration.resolve(),
        rollout_path=args.rollout.resolve(),
        frozen_release=args.frozen_release.resolve(),
        final_test_gate=args.final_test_gate.resolve(),
        canonical_output=args.canonical_output.resolve(),
        map_output=args.map_output.resolve(),
    )
    print(
        f"[rate10k] Map export complete: outputs={len(report['outputs'])}, "
        f"pairs={report['pair_count']:,}, tiles={report['tiles']:,}, "
        f"canonical={report['canonical_predictions']}, "
        f"maps={report['contact_maps']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

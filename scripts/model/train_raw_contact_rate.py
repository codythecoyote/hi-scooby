#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from raw_contact_ranker.common import load_config
from raw_contact_ranker.exact_rate import train_exact_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/raw_contact_ranker_10kb.yaml",
    )
    parser.add_argument(
        "--feature-set",
        choices=("annotations", "alphagenome", "combined"),
        required=True,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    epochs = (
        args.epochs
        if args.epochs is not None
        else int(config["model"]["epochs"])
    )
    fold_label = "all training chromosomes" if args.fold is None else f"fold {args.fold}"
    print(
        f"[rate10k] Training {args.feature_set} shared topology on "
        f"{fold_label} for up to {epochs} epochs (seed={seed})",
        flush=True,
    )
    report = train_exact_rate(
        config,
        output_dir=args.output_dir.resolve(),
        feature_set=args.feature_set,
        seed=seed,
        epochs=epochs,
        fold=args.fold,
        resume_from=args.resume_from.resolve() if args.resume_from else None,
    )
    print(
        f"[rate10k] Shared topology training complete: "
        f"best_epoch={report['best_epoch']}, "
        f"best_validation_loss={report['best_validation_loss']:.6g}, "
        f"output={args.output_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()

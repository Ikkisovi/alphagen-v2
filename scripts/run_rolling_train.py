#!/usr/bin/env python3
"""
Run rolling window training for AlphaGen factors.

This script orchestrates the full pipeline:
1. Generate rolling time windows
2. Train AlphaGen on each window
3. Export results to Lean strategies

Usage:
    # Train all windows
    python scripts/run_rolling_train.py

    # Train specific windows
    python scripts/run_rolling_train.py --start-window 0 --end-window 3

    # Use custom config
    python scripts/run_rolling_train.py --config my_config.py

    # Quick test with fewer training steps
    python scripts/run_rolling_train.py --steps 1000 --end-window 1
"""

import argparse
import sys
from pathlib import Path

# Add alphagen to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen_lean.rolling_config import RollingConfig
from alphagen_lean.rolling_trainer import RollingTrainer
from alphagen_lean.lean_exporter import LeanExporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run rolling window training for AlphaGen factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--start-window',
        type=int,
        default=0,
        help='Start training from this window index (default: 0)'
    )

    parser.add_argument(
        '--end-window',
        type=int,
        default=None,
        help='End training at this window index (default: all windows)'
    )

    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Override training steps (default: from config)'
    )

    parser.add_argument(
        '--pool-capacity',
        type=int,
        default=None,
        help='Override pool capacity (default: from config)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Override output directory (default: from config)'
    )

    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Skip training, only export existing results to Lean'
    )

    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip Lean export, only train'
    )

    parser.add_argument(
        '--export-objectstore',
        action='store_true',
        help='Also export factors for ObjectStore (dynamic loading)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device (e.g., "cpu", "cuda:0")'
    )

    parser.add_argument(
        '--resolution',
        type=str,
        default='daily',
        choices=['minute', 'daily'],
        help='Data resolution (default: daily)'
    )

    parser.add_argument(
        '--train-strategy',
        type=str,
        default='dual_stage',
        choices=['single', 'dual_stage'],
        help='Training strategy (default: dual_stage)'
    )

    parser.add_argument(
        '--price-steps',
        type=int,
        default=None,
        help='Steps for price-only stage (dual_stage mode only)'
    )

    parser.add_argument(
        '--fundamental-steps',
        type=int,
        default=None,
        help='Steps for fundamental stage (dual_stage mode only)'
    )

    parser.add_argument(
        '--evaluate-deploy',
        action='store_true',
        help='Enable deploy-period IC evaluation (default disabled for forward prediction scenarios)'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    print(f"\n{'='*80}")
    print("Rolling Window AlphaGen Training")
    print(f"{'='*80}\n")

    config = RollingConfig()

    # Apply command-line overrides
    if args.steps is not None:
        config.train_steps = args.steps
        print(f"Override: train_steps = {args.steps}")

    if args.pool_capacity is not None:
        config.pool_capacity = args.pool_capacity
        print(f"Override: pool_capacity = {args.pool_capacity}")

    if args.output_dir is not None:
        config.output_dir = args.output_dir
        print(f"Override: output_dir = {args.output_dir}")

    if args.device is not None:
        import torch
        config.device = torch.device(args.device)
        print(f"Override: device = {args.device}")

    # New overrides for daily data and dual-stage training
    if args.resolution:
        config.data_resolution = args.resolution
        print(f"Override: data_resolution = {args.resolution}")

    if args.train_strategy:
        config.train_strategy = args.train_strategy
        print(f"Override: train_strategy = {args.train_strategy}")

    if args.price_steps is not None:
        config.price_stage_steps = args.price_steps
        print(f"Override: price_stage_steps = {args.price_steps}")

    if args.fundamental_steps is not None:
        config.fundamental_stage_steps = args.fundamental_steps
        print(f"Override: fundamental_stage_steps = {args.fundamental_steps}")

    if args.evaluate_deploy:
        config.evaluate_deploy = True
        print("Override: evaluate_deploy = True")

    # Print configuration summary
    print(f"\nConfiguration:")
    print(f"  Data resolution: {config.data_resolution}")
    print(f"  Data path: {config.data_path}")
    print(f"  Symbols: {len(config.symbols)}")
    print(f"  Training strategy: {config.train_strategy}")
    if config.train_strategy == "dual_stage":
        print(f"  Price stage steps: {config.price_stage_steps}")
        print(f"  Fundamental stage steps: {config.fundamental_stage_steps}")
    else:
        print(f"  Training steps: {config.train_steps}")
    print(f"  Pool capacity: {config.pool_capacity}")
    print(f"  Device: {config.device}")
    print(f"  Deploy evaluation: {config.evaluate_deploy}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Lean output directory: {config.lean_project_dir}")
    print()

    # Training phase
    if not args.export_only:
        print(f"{'='*80}")
        print("Phase 1: Training")
        print(f"{'='*80}\n")

        trainer = RollingTrainer(config)

        results = trainer.train_all_windows(
            start_window=args.start_window,
            end_window=args.end_window
        )

        print(f"\nTraining completed! {len(results)} windows trained.")

    # Export phase
    if not args.no_export:
        print(f"\n{'='*80}")
        print("Phase 2: Export to Lean")
        print(f"{'='*80}\n")

        exporter = LeanExporter(config)

        if args.export_only:
            # Load results from output directory
            import json
            summary_path = config.output_dir / "training_summary.json"

            if not summary_path.exists():
                print(f"ERROR: Training summary not found: {summary_path}")
                print("Please run training first or specify correct --output-dir")
                return 1

            with open(summary_path, 'r') as f:
                summary = json.load(f)
                results = summary['windows']

            print(f"Loaded {len(results)} windows from {summary_path}")

        exporter.export_all_windows(results)

        # Export for ObjectStore if requested
        if args.export_objectstore:
            print(f"\n{'='*80}")
            print("Phase 3: Export for ObjectStore")
            print(f"{'='*80}\n")
            exporter.export_all_factors_for_objectstore(results)

    # Summary
    print(f"\n{'='*80}")
    print("Pipeline Complete!")
    print(f"{'='*80}")

    if not args.export_only:
        print(f"\nTraining results saved to: {config.output_dir}")

    if not args.no_export:
        print(f"Lean strategies exported to: {config.lean_project_dir}")
        print(f"\nTo backtest a strategy:")
        print(f"  cd {config.lean_project_dir}/window_YYYY_MM")
        print(f"  lean backtest")

    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

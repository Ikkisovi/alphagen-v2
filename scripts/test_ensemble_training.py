#!/usr/bin/env python3
"""
Test script for ensemble training with merged OHLCV + fundamental data.

This script performs a quick test run of the ensemble training pipeline:
1. Verifies data availability
2. Runs a short training session (reduced episodes)
3. Validates outputs

Usage:
    python scripts/test_ensemble_training.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from datetime import datetime


def verify_data_availability(config: dict) -> bool:
    """Verify all required data files exist."""
    print("=" * 80)
    print("DATA AVAILABILITY CHECK")
    print("=" * 80)

    checks = []

    # Check merged data
    merged_parquet = Path("data/merged/merged_data.parquet")
    checks.append(("Merged OHLCV+Fundamentals", merged_parquet.exists(), merged_parquet))

    # Check fundamental data
    fund_parquet = Path("lean_project/data/fundamentals/fundamentals.parquet")
    checks.append(("Fundamental data", fund_parquet.exists(), fund_parquet))

    # Check OHLCV directory
    ohlcv_dir = Path("e:/factor/lean_project/data/equity/usa/daily")
    has_ohlcv = ohlcv_dir.exists() and len(list(ohlcv_dir.glob("*.zip"))) > 0
    checks.append(("OHLCV zip files", has_ohlcv, ohlcv_dir))

    # Print results
    all_ok = True
    for name, exists, path in checks:
        status = "✓ OK" if exists else "✗ MISSING"
        print(f"  {status}: {name}")
        print(f"         {path}")
        if not exists:
            all_ok = False

    print()
    return all_ok


def create_test_config(base_config_path: Path) -> dict:
    """Create a test configuration with reduced episodes for quick validation."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)

    # Reduce training episodes for quick test
    original_stage1 = config['training']['stage1_technical']['max_episodes']

    config['training']['stage1_technical']['max_episodes'] = 50
    config['training']['stage1_technical']['min_episodes'] = 30
    config['training']['stage1_technical']['early_stopping_patience'] = 10

    # Reduce pool capacity
    config['training']['pool_capacity_per_window'] = 15
    config['ensemble']['final_capacity'] = 30

    print(f"  Stage 1 episodes: {original_stage1} → {config['training']['stage1_technical']['max_episodes']} (test mode)")
    print(f"  Pool capacity: 25 → {config['training']['pool_capacity_per_window']} (test mode)")
    print(f"  Final capacity: 45 → {config['ensemble']['final_capacity']} (test mode)")

    # Update output paths for test
    config['output']['base_dir'] = 'output/test_ensemble'
    config['output']['pool_12m'] = 'output/test_ensemble/pool_12m.json'
    config['output']['pool_6m'] = 'output/test_ensemble/pool_6m.json'
    config['output']['pool_3m'] = 'output/test_ensemble/pool_3m.json'
    config['output']['final_ensemble'] = 'output/test_ensemble/ensemble_pool_test.json'
    config['output']['training_log'] = 'output/test_ensemble/test_training.log'
    config['output']['performance_report'] = 'output/test_ensemble/test_report.txt'
    config['output']['metrics_file'] = 'output/test_ensemble/test_metrics.json'

    # Disable cache for test
    config['cache']['enabled'] = False

    print(f"  Output directory: {config['output']['base_dir']}")
    print()

    return config


def run_test_training(config: dict):
    """Run the ensemble training with test configuration."""
    print("=" * 80)
    print("STARTING TEST TRAINING RUN")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Import training script
    from scripts.train_ensemble import main as train_main

    # Save test config
    test_config_path = Path('config/test_ensemble_config.yaml')
    test_config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Test config saved to: {test_config_path}")
    print()

    # Run training
    try:
        train_main(
            config_file=str(test_config_path),
            skip_training=False,
            seed=42
        )
        print("\n" + "=" * 80)
        print("TEST TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_outputs(config: dict):
    """Verify that expected output files were created."""
    print("\n" + "=" * 80)
    print("OUTPUT VERIFICATION")
    print("=" * 80)

    expected_files = [
        config['output']['pool_12m'],
        config['output']['pool_6m'],
        config['output']['pool_3m'],
        config['output']['final_ensemble'],
        config['output']['training_log'],
        config['output']['performance_report'],
    ]

    all_ok = True
    for file_path in expected_files:
        path = Path(file_path)
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = f"({path.stat().st_size / 1024:.1f} KB)" if exists else ""

        print(f"  {status} {path} {size}")

        if not exists:
            all_ok = False

    print()
    return all_ok


def print_summary(config: dict):
    """Print a summary of the test results."""
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    ensemble_file = Path(config['output']['final_ensemble'])
    if ensemble_file.exists():
        with open(ensemble_file, 'r') as f:
            ensemble = json.load(f)

        print(f"  Final ensemble size: {len(ensemble['exprs'])} factors")
        print(f"  Top 5 factors by weight:")

        weights = ensemble['weights']
        exprs = ensemble['exprs']

        # Sort by absolute weight
        import numpy as np
        sorted_indices = np.argsort(-np.abs(weights))[:5]

        for i, idx in enumerate(sorted_indices, 1):
            print(f"    {i}. Weight: {weights[idx]:+.4f}")
            print(f"       {exprs[idx][:80]}{'...' if len(exprs[idx]) > 80 else ''}")

    report_file = Path(config['output']['performance_report'])
    if report_file.exists():
        print(f"\n  Full report available at: {report_file}")

    print()


def main():
    """Main test orchestration."""
    print("\n" + "=" * 80)
    print("ENSEMBLE TRAINING TEST SUITE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    # Load base config
    base_config_path = Path('config/ensemble_config.yaml')
    if not base_config_path.exists():
        print(f"ERROR: Base config not found: {base_config_path}")
        return 1

    # Verify data
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    if not verify_data_availability(base_config):
        print("ERROR: Required data files are missing!")
        print("Please run:")
        print("  1. python scripts/build_fundamental_dataset.py")
        print("  2. python scripts/merge_ohlcv_fundamentals.py")
        return 1

    # Create test config
    test_config = create_test_config(base_config_path)

    # Run training
    success = run_test_training(test_config)

    if not success:
        print("\nTest training failed. See error messages above.")
        return 1

    # Verify outputs
    if not verify_outputs(test_config):
        print("ERROR: Some expected output files are missing!")
        return 1

    # Print summary
    print_summary(test_config)

    print("=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)
    print("\nThe ensemble training pipeline is ready for production use.")
    print("To run full training, use:")
    print("  python scripts/train_ensemble.py --config config/ensemble_config.yaml")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())

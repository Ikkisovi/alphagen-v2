"""
Test script for ensemble alpha mining system.

This script validates the ensemble training infrastructure without
running expensive full training. It tests:
- Configuration loading
- Cache management
- Pool creation and merging
- Data loading
- Expression parsing

Usage:
    python scripts/test_ensemble.py
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import yaml
import torch
import numpy as np

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, FeatureType, initialize_qlib
from alphagen_generic.operators import Operators


def test_config_loading():
    """Test configuration file loading."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Loading")
    print("="*80)

    config_file = "config/ensemble_config.yaml"

    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        print(f"✓ Successfully loaded config from {config_file}")

        # Validate required sections
        required_sections = [
            'time_windows',
            'universe',
            'training',
            'fundamental_features',
            'ensemble',
            'cache',
            'output',
            'logging'
        ]

        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
            print(f"  ✓ Found section: {section}")

        # Validate time windows
        assert 'train_12m' in config['time_windows']
        assert 'train_6m' in config['time_windows']
        assert 'train_3m' in config['time_windows']
        assert 'validation' in config['time_windows']
        print(f"  ✓ All time windows configured")

        print("\n✅ Configuration loading test PASSED\n")
        return config

    except Exception as e:
        print(f"\n❌ Configuration loading test FAILED: {e}\n")
        raise


def test_data_loading(config):
    """Test data loading with minimal time range."""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)

    try:
        # Initialize qlib
        print("Initializing qlib...")
        initialize_qlib()
        print("✓ Qlib initialized")

        # Load a small data sample (1 month)
        universe = config['universe']['base_instrument']
        print(f"Loading data for universe: {universe}")

        device = torch.device('cpu')  # Use CPU for testing

        data = StockData(
            instrument=universe,
            start_time="2024-01-01",
            end_time="2024-01-31",
            device=device
        )

        print(f"✓ Data loaded: shape {data.data.shape}")
        print(f"  - Days: {data.n_days}")
        print(f"  - Stocks: {data.n_stocks}")
        print(f"  - Features: {len(data._features)}")

        # Test fundamental features
        print("\nTesting fundamental features...")
        all_features = list(FeatureType)
        data_with_fundamentals = StockData(
            instrument=universe,
            start_time="2024-01-01",
            end_time="2024-01-31",
            features=all_features,
            device=device
        )

        print(f"✓ Data loaded with {len(all_features)} features")

        print("\n✅ Data loading test PASSED\n")
        return data

    except Exception as e:
        print(f"\n❌ Data loading test FAILED: {e}\n")
        raise


def test_pool_creation(config, data):
    """Test alpha pool creation and basic operations."""
    print("\n" + "="*80)
    print("TEST 3: Alpha Pool Creation")
    print("="*80)

    try:
        # Create target
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -20) / close - 1
        print("✓ Target expression created")

        # Create calculator
        calculator = QLibStockDataCalculator(data, target)
        print("✓ Calculator created")

        # Create pool
        device = torch.device('cpu')
        pool = MseAlphaPool(
            capacity=10,
            calculator=calculator,
            ic_lower_bound=0.01,
            l1_alpha=0.005,
            device=device
        )
        print(f"✓ Pool created with capacity {pool.capacity}")

        # Test adding expressions manually
        test_exprs = [
            close,
            Feature(FeatureType.VOLUME),
            Feature(FeatureType.VWAP)
        ]

        print(f"\nTesting expression addition...")
        pool.force_load_exprs(test_exprs)
        print(f"✓ Loaded {pool.size} expressions into pool")

        # Test pool to JSON
        pool_dict = pool.to_json_dict()
        print(f"✓ Converted pool to JSON: {len(pool_dict['exprs'])} expressions")

        print("\n✅ Pool creation test PASSED\n")
        return pool

    except Exception as e:
        print(f"\n❌ Pool creation test FAILED: {e}\n")
        raise


def test_expression_parsing():
    """Test expression parsing and serialization."""
    print("\n" + "="*80)
    print("TEST 4: Expression Parsing")
    print("="*80)

    try:
        parser = ExpressionParser(Operators)

        test_expressions = [
            "$close",
            "$volume",
            "Div($close, $volume)",
            "Sub($high, $low)",
            "Greater($close, $open)"
        ]

        print("Testing expression parsing...")
        for expr_str in test_expressions:
            expr = parser.parse(expr_str)
            serialized = str(expr)
            print(f"  ✓ {expr_str} -> {serialized}")

            # Test round-trip
            expr2 = parser.parse(serialized)
            assert str(expr2) == serialized, f"Round-trip failed for {expr_str}"

        print("\n✅ Expression parsing test PASSED\n")

    except Exception as e:
        print(f"\n❌ Expression parsing test FAILED: {e}\n")
        raise


def test_pool_merging(config):
    """Test merging multiple pools."""
    print("\n" + "="*80)
    print("TEST 5: Pool Merging")
    print("="*80)

    try:
        # Create mock pool files
        print("Creating mock pool files...")

        temp_dir = Path(tempfile.mkdtemp())
        print(f"Using temp directory: {temp_dir}")

        mock_pools = {
            '12m': {
                'exprs': ['$close', 'Div($close, $volume)', 'Sub($high, $low)'],
                'weights': [0.5, 0.3, 0.2]
            },
            '6m': {
                'exprs': ['$volume', 'Div($close, $volume)', '$vwap'],
                'weights': [0.4, 0.4, 0.2]
            },
            '3m': {
                'exprs': ['$high', '$low', 'Sub($high, $low)'],
                'weights': [0.3, 0.3, 0.4]
            }
        }

        # Save mock pools
        for name, pool_data in mock_pools.items():
            pool_file = temp_dir / f'pool_{name}.json'
            with open(pool_file, 'w') as f:
                json.dump(pool_data, f)
            print(f"  ✓ Created {pool_file}")

        # Load and merge
        print("\nMerging pools...")
        all_expressions = []
        for name, pool_data in mock_pools.items():
            all_expressions.extend(pool_data['exprs'])

        print(f"✓ Collected {len(all_expressions)} expressions")

        # Deduplicate
        unique_expressions = list(set(all_expressions))
        print(f"✓ After deduplication: {len(unique_expressions)} unique expressions")

        # Parse all
        parser = ExpressionParser(Operators)
        parsed = [parser.parse(expr) for expr in unique_expressions]
        print(f"✓ Successfully parsed all {len(parsed)} expressions")

        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"✓ Cleaned up temp directory")

        print("\n✅ Pool merging test PASSED\n")

    except Exception as e:
        print(f"\n❌ Pool merging test FAILED: {e}\n")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def test_cache_management(config):
    """Test cache management functionality."""
    print("\n" + "="*80)
    print("TEST 6: Cache Management")
    print("="*80)

    try:
        from scripts.train_ensemble import CacheManager
        import logging

        logger = logging.getLogger('test')
        logger.setLevel(logging.INFO)

        cache_manager = CacheManager(config, logger)
        print("✓ CacheManager created")

        # Test universe fingerprint
        universe_config = config['universe']
        fingerprint = cache_manager.get_universe_fingerprint(universe_config)
        print(f"✓ Universe fingerprint: {fingerprint}")

        # Test cache freshness
        temp_file = Path(tempfile.mktemp())
        temp_file.touch()

        is_fresh = cache_manager.is_cache_fresh(temp_file, freshness_days=30)
        print(f"✓ Cache freshness check: {is_fresh}")

        temp_file.unlink()

        print("\n✅ Cache management test PASSED\n")

    except Exception as e:
        print(f"\n❌ Cache management test FAILED: {e}\n")
        raise


def test_fundamental_features():
    """Test that all fundamental features are accessible."""
    print("\n" + "="*80)
    print("TEST 7: Fundamental Features")
    print("="*80)

    try:
        from alphagen_generic import features

        fundamental_features = [
            'pe_ratio', 'pb_ratio', 'ps_ratio',
            'ev_to_ebitda', 'ev_to_revenue', 'ev_to_fcf',
            'earnings_yield', 'fcf_yield', 'sales_yield', 'dividend_yield',
            'revenue_growth', 'earnings_growth', 'book_value_growth',
            'debt_to_assets', 'debt_to_equity', 'current_ratio', 'quick_ratio',
            'roe', 'roa', 'roic',
            'gross_margin', 'operating_margin', 'net_margin'
        ]

        print("Checking fundamental features...")
        for feat_name in fundamental_features:
            assert hasattr(features, feat_name), f"Feature {feat_name} not found"
            feat = getattr(features, feat_name)
            print(f"  ✓ {feat_name}: {feat}")

        print(f"\n✓ All {len(fundamental_features)} fundamental features accessible")

        print("\n✅ Fundamental features test PASSED\n")

    except Exception as e:
        print(f"\n❌ Fundamental features test FAILED: {e}\n")
        raise


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENSEMBLE TRAINING SYSTEM - TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")

    try:
        # Test 1: Config loading
        config = test_config_loading()

        # Test 2: Data loading
        data = test_data_loading(config)

        # Test 3: Pool creation
        pool = test_pool_creation(config, data)

        # Test 4: Expression parsing
        test_expression_parsing()

        # Test 5: Pool merging
        test_pool_merging(config)

        # Test 6: Cache management
        test_cache_management(config)

        # Test 7: Fundamental features
        test_fundamental_features()

        # Summary
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print(f"Completed: {datetime.now().isoformat()}")
        print("\nThe ensemble training system is ready to use!")
        print("Run: python scripts/train_ensemble.py")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print(f"TESTS FAILED ❌")
        print("="*80)
        print(f"Error: {e}")
        print("="*80 + "\n")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

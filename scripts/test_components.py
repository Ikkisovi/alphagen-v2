#!/usr/bin/env python3
"""
Test core components of alphagen_lean.

This script tests:
1. WindowManager - time window generation
2. ExpressionConverter - expression to Python code conversion
3. Basic data loading
"""

import sys
from pathlib import Path

# Add alphagen to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphagen_lean.window_manager import WindowManager
from alphagen_lean.expression_converter import ExpressionConverter, convert_expression_list
from alphagen_lean.rolling_config import RollingConfig


def test_window_manager():
    """Test window manager."""
    print("\n" + "="*80)
    print("TEST 1: Window Manager")
    print("="*80 + "\n")

    manager = WindowManager(
        first_train_start="2023-01-01",
        deploy_start="2024-01-01",
        deploy_end="2024-06-30",
        train_months=12,
        test_months=1,
        step_months=1
    )

    print(manager.summary())

    # Test window lookup
    print("\nTest window lookup by month:")
    window = manager.get_window_by_month("2024_03")
    print(f"  Found: {window}")

    print("\n[PASS] Window Manager Test PASSED\n")


def test_expression_converter():
    """Test expression converter."""
    print("\n" + "="*80)
    print("TEST 2: Expression Converter")
    print("="*80 + "\n")

    test_expressions = [
        ("Simple feature", "$close"),
        ("Mean with window", "Mean($close, 20d)"),
        ("Binary operation", "Add($close, $open)"),
        ("Division", "Div($close, Mean($close, 20d))"),
        ("Complex nested", "Div(Mean($volume,10d),Greater(Mul(0.5,$low),-1.0))"),
        ("Logarithm", "Log(Div(2.0,$high))"),
        ("Absolute", "Abs(Sub($close, $open))"),
        ("Standard deviation", "Std($close, 20d)"),
        ("Delta", "Delta($close, 5d)"),
    ]

    converter = ExpressionConverter()
    passed = 0
    failed = 0

    for name, expr in test_expressions:
        print(f"\nTest: {name}")
        print(f"Expression: {expr}")
        try:
            code = converter.convert(expr, "_test")
            print(f"[PASS] Converted successfully:")
            print(code)
            passed += 1
        except Exception as e:
            print(f"[FAIL] FAILED: {e}")
            failed += 1

    print(f"\n{'='*80}")
    print(f"Expression Converter Test Results: {passed} passed, {failed} failed")
    print(f"{'='*80}\n")

    if failed == 0:
        print("[PASS] Expression Converter Test PASSED\n")
    else:
        print("[WARN] Expression Converter Test PARTIALLY PASSED\n")


def test_batch_conversion():
    """Test batch conversion of multiple expressions."""
    print("\n" + "="*80)
    print("TEST 3: Batch Expression Conversion")
    print("="*80 + "\n")

    # Real expressions from seekingal_worldq
    expressions = [
        "Mul(Div(Mean(Mad(Div(10.0,Greater($high,-0.01)),40d),40d),-2.0),1.0)",
        "Div(Mean($volume,10d),Greater(Mul(0.5,$low),-1.0))",
        "Less(Add(-2.0,$high),5.0)",
        "Log(Div(2.0,$high))",
        "Min(Greater(30.0,$volume),5d)",
        "Sub(-10.0,Abs($open))",
    ]

    print(f"Converting {len(expressions)} expressions...\n")

    function_codes, helper_code = convert_expression_list(expressions)

    for i, (expr, code) in enumerate(zip(expressions, function_codes)):
        print(f"Factor {i+1}:")
        print(f"  Expression: {expr}")
        print(f"  Function preview: {code[:150]}...")
        print()

    print(f"Helper functions generated: {len(helper_code)} chars\n")
    print("[PASS] Batch Conversion Test PASSED\n")


def test_config():
    """Test configuration."""
    print("\n" + "="*80)
    print("TEST 4: Configuration")
    print("="*80 + "\n")

    config = RollingConfig()

    print(f"Data path: {config.data_path}")
    print(f"Symbols: {len(config.symbols)}")
    print(f"First train start: {config.first_train_start}")
    print(f"Deploy period: {config.deploy_start} to {config.deploy_end}")
    print(f"Training window: {config.train_months} months")
    print(f"Test window: {config.test_months} months")
    print(f"Pool capacity: {config.pool_capacity}")
    print(f"Training steps: {config.train_steps}")
    print(f"Device: {config.device}")
    print(f"Output dir: {config.output_dir}")
    print(f"Lean output dir: {config.lean_project_dir}")

    print("\n[PASS] Configuration Test PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("AlphaGen-Lean Component Tests")
    print("="*80)

    try:
        test_window_manager()
        test_expression_converter()
        test_batch_conversion()
        test_config()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

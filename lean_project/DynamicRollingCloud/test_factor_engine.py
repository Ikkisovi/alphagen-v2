"""Test script for validating factor_engine.py expression parsing."""

import sys
import numpy as np
import pandas as pd

# Mock algorithm class for testing
class MockAlgorithm:
    def Debug(self, msg):
        print(f"[DEBUG] {msg}")

# Import factor engine
from factor_engine import ExpressionConverter, FactorEngine, convert_expression_list

def test_expression_conversion():
    """Test individual expression conversion."""
    print("="*60)
    print("Testing Expression Conversion")
    print("="*60)

    # Test expressions from the error log
    test_expressions = [
        "$sales_yield",
        "Div(Sub(Abs(Log(Greater(-5.0,Ref($low,20d)))),2.0),$vwap)",
        "Var(Less($volume,Div($volume,$high)),20d)",
        "Log(Mean($vwap,5d))",
        "Log($close)",
        "Mul(Div($low,Mul(Log($ev_to_fcf),-30.0)),Mul(Greater($low,-10.0),$ps_ratio))",
        "Log(Greater(Add(Greater($close,-0.01),5.0),30.0))",
        "Greater($pb_ratio,-5.0)",
        "$earnings_yield",
        "Greater(WMA(Greater($open,1.0),10d),5.0)",
    ]

    converter = ExpressionConverter()

    for i, expr in enumerate(test_expressions, 1):
        print(f"\nTest {i}: {expr}")
        try:
            func_code = converter.convert(expr, f"_test_{i}")
            print(f"[OK] SUCCESS")
            print(f"  Generated code (first 100 chars): {func_code[:100]}...")
        except Exception as e:
            print(f"[FAIL] FAILED: {e}")

    return True

def test_factor_engine():
    """Test FactorEngine compilation and evaluation."""
    print("\n" + "="*60)
    print("Testing FactorEngine")
    print("="*60)

    # Create mock algorithm
    algo = MockAlgorithm()
    engine = FactorEngine(algo)

    # Simple test expressions
    expressions = [
        "$close",
        "Mean($close, 5d)",
        "Add($close, $open)",
        "Log($close)",
    ]

    print(f"\nCompiling {len(expressions)} expressions...")
    try:
        engine.compile(expressions)
        print("[OK] Compilation successful")
    except Exception as e:
        print(f"[X] Compilation failed: {e}")
        return False

    # Create test data bundle
    print("\nTesting evaluation with sample data...")
    history_bundle = {
        'close': np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        'open': np.array([99.0, 100.5, 101.5, 102.5, 103.5, 104.5]),
        'high': np.array([101.0, 102.0, 103.0, 104.0, 105.0, 106.0]),
        'low': np.array([98.0, 99.0, 100.0, 101.0, 102.0, 103.0]),
        'volume': np.array([1000000, 1100000, 1200000, 1300000, 1400000, 1500000]),
        'vwap': np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]),
        # Fundamental fields (will be NaN for now)
        'pe_ratio': np.full(6, np.nan),
        'pb_ratio': np.full(6, np.nan),
        'ps_ratio': np.full(6, np.nan),
        'ev_to_ebitda': np.full(6, np.nan),
        'ev_to_revenue': np.full(6, np.nan),
        'ev_to_fcf': np.full(6, np.nan),
        'earnings_yield': np.full(6, np.nan),
        'fcf_yield': np.full(6, np.nan),
        'sales_yield': np.full(6, np.nan),
        'forward_pe_ratio': np.full(6, np.nan),
        'market_cap': np.full(6, np.nan),
        'shares_outstanding': np.full(6, np.nan),
        'turnover': np.full(6, np.nan),
    }

    try:
        results = engine.evaluate(history_bundle)
        print(f"[OK] Evaluation successful")
        print(f"  Results: {results}")

        # Validate results
        if len(results) != len(expressions):
            print(f"[X] Wrong number of results: expected {len(expressions)}, got {len(results)}")
            return False

        # Check if results are reasonable
        for i, (expr, result) in enumerate(zip(expressions, results)):
            if np.isfinite(result):
                print(f"  Factor {i+1} ({expr}): {result:.4f}")
            else:
                print(f"  Factor {i+1} ({expr}): {result} (NaN/Inf)")

    except Exception as e:
        print(f"[X] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

def test_complex_expressions():
    """Test the exact expressions from the error log."""
    print("\n" + "="*60)
    print("Testing Complex Expressions from Error Log")
    print("="*60)

    algo = MockAlgorithm()
    engine = FactorEngine(algo)

    expressions = [
        "$sales_yield",
        "Div(Sub(Abs(Log(Greater(-5.0,Ref($low,20d)))),2.0),$vwap)",
        "Var(Less($volume,Div($volume,$high)),20d)",
        "Log(Mean($vwap,5d))",
        "Log($close)",
    ]

    print(f"\nCompiling {len(expressions)} complex expressions...")
    try:
        engine.compile(expressions)
        print("[OK] Compilation successful")
    except Exception as e:
        print(f"[X] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create test data with enough history
    n_days = 120
    history_bundle = {
        'close': np.linspace(100, 110, n_days),
        'open': np.linspace(99, 109, n_days),
        'high': np.linspace(101, 111, n_days),
        'low': np.linspace(98, 108, n_days),
        'volume': np.random.uniform(1000000, 2000000, n_days),
        'vwap': np.linspace(100, 110, n_days),
        'pe_ratio': np.full(n_days, 15.0),
        'pb_ratio': np.full(n_days, 2.5),
        'ps_ratio': np.full(n_days, 3.0),
        'ev_to_ebitda': np.full(n_days, 10.0),
        'ev_to_revenue': np.full(n_days, 5.0),
        'ev_to_fcf': np.full(n_days, 12.0),
        'earnings_yield': np.full(n_days, 0.06),
        'fcf_yield': np.full(n_days, 0.05),
        'sales_yield': np.full(n_days, 1.0/3.0),  # 1/PS ratio
        'forward_pe_ratio': np.full(n_days, 14.0),
        'market_cap': np.full(n_days, 1e9),
        'shares_outstanding': np.full(n_days, 1e7),
        'turnover': np.full(n_days, 0.01),
    }

    print("\nEvaluating expressions...")
    try:
        results = engine.evaluate(history_bundle)
        print("[OK] Evaluation successful")

        for i, (expr, result) in enumerate(zip(expressions, results), 1):
            status = "[OK]" if np.isfinite(result) else "[!]"
            result_str = f"{result:.6f}" if np.isfinite(result) else str(result)
            print(f"  {status} Factor {i}: {result_str}")
            print(f"     Expression: {expr[:60]}{'...' if len(expr) > 60 else ''}")

    except Exception as e:
        print(f"[X] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    print("Factor Engine Test Suite")
    print("=" * 60)

    # Run tests
    test1 = test_expression_conversion()
    test2 = test_factor_engine()
    test3 = test_complex_expressions()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Expression Conversion: {'[OK] PASS' if test1 else '[X] FAIL'}")
    print(f"Factor Engine Basic: {'[OK] PASS' if test2 else '[X] FAIL'}")
    print(f"Complex Expressions: {'[OK] PASS' if test3 else '[X] FAIL'}")

    if test1 and test2 and test3:
        print("\n[OK] All tests passed!")
        sys.exit(0)
    else:
        print("\n[X] Some tests failed!")
        sys.exit(1)

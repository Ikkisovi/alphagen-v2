# Debug Findings v0.11 - Fundamental Data Issue

**Date**: 2025-11-01
**Status**: Critical issue identified

## Problem Summary

Enhanced debug logging revealed that **Factor 1 is completely ineffective** because fundamental data (pb_ratio, etc.) is constant over time.

## Detailed Findings

### Factor 1: Mad($pb_ratio,5d) - ALL ZEROS

**Raw factor values (Rebalance #1):**
```
MU:   ['0.0000000000e+00', '-8.7389667603e-10']
TTMI: ['0.0000000000e+00', '-9.8998121014e-09']
CDE:  ['0.0000000000e+00', '-3.5611770259e-09']
```

**Normalization stats:**
```
Factor 1: mean=0.0000000000e+00, std=0.0000000000e+00, valid=40/40
  Raw values (first 5 stocks): all 0.0000000000e+00
  WARNING: std < 1e-8, setting normalized values to 0
```

**Root Cause:**
Mean Absolute Deviation (MAD) of `$pb_ratio` over 5 days is 0 because **pb_ratio is constant** across all 60 daily bars. The CSV fundamental data has the same value repeated for every day.

### Factor 2: Complex Factor - WORKING (but tiny values)

**Raw factor values:**
```
MU:   -8.7389667603e-10
TTMI: -9.8998121014e-09
CDE:  -3.5611770259e-09
```

**Normalization stats:**
```
Factor 2: mean=-1.7753604136e-08, std=2.2366184536e-08, valid=32/40
```

This factor has differentiation (std=2.24e-08) and produces all the final rankings.

### Fundamental Data Loading Status

**✅ Fundamentals ARE loading correctly:**
```
[DEBUG] History for MU (60 bars):
    OHLCV fields: open=True, close=True
    Fundamental pe_ratio: 60/60 valid, last=2.666400e+01
    Fundamental pb_ratio: 60/60 valid, last=4.193940e+00
    Fundamental ps_ratio: 60/60 valid, last=6.091220e+00

[DEBUG] History for TTMI (60 bars):
    OHLCV fields: open=True, close=True
    Fundamental pe_ratio: 60/60 valid, last=6.422470e+01
    Fundamental pb_ratio: 60/60 valid, last=3.607620e+00
    Fundamental ps_ratio: 60/60 valid, last=2.255740e+00
```

Different stocks have different values (MU pb_ratio=4.19, TTMI pb_ratio=3.61), BUT each stock's value is constant across all 60 days.

### Final Rankings

**Rankings come ENTIRELY from Factor 2** (the complex factor):
```
Top:
  1. POWL = +0.1472
  2. WLDN = +0.1272
  3. AGX  = +0.1155

Bottom:
 39. KGC  = -0.0373
 40. UBER = -0.0378
```

Formula: `final_score = 0.1823 * factor1_normalized + (-0.0491) * factor2_normalized`
Since factor1_normalized=0 for all stocks: `final_score = (-0.0491) * factor2_normalized`

## Root Cause Analysis

### Why Fundamental Data is Constant

The CSV files in `lean_project/data/equity/usa/custom/manual_fundamentals/` likely contain **quarterly fundamental data** repeated for every day:

```csv
2022-01-01,4.19394
2022-01-02,4.19394  ← Same value
2022-01-03,4.19394  ← Same value
...
2022-03-31,4.19394  ← Until next quarter
2022-04-01,4.25000  ← New quarterly value
```

This is typical for fundamental data which updates quarterly, not daily.

### Why This Breaks Factor 1

**Mad($pb_ratio,5d)** calculates Mean Absolute Deviation over 5 days:
- If pb_ratio[t] = pb_ratio[t-1] = pb_ratio[t-2] = ... (constant)
- Then MAD = mean(|values - mean|) = mean(|4.19 - 4.19|) = 0

Any time-series operator (Mad, Std, Delta, etc.) on constant data returns 0.

### Why Factor 2 Still Works

Factor 2 uses **cross-sectional operators** and combinations:
```
Div(-1.0, Sub($ev_to_ebitda, Min(Sub(Abs(WMA($ps_ratio,40d)), $shares_outstanding), 10d)))
```

- `WMA($ps_ratio, 40d)`: Even though ps_ratio is constant, WMA returns the same value
- `Sub(Abs(WMA(...)), $shares_outstanding)`: Cross-stock calculation using shares_outstanding
- Different stocks have different shares_outstanding → produces differentiation
- The complexity creates **very small** but non-zero differences (1e-8 to 1e-10)

## Solutions

### Option 1: Use Only Cross-Sectional Fundamental Factors

Instead of:
- `Mad($pb_ratio,5d)` ❌ (time-series, returns 0)

Use:
- `$pb_ratio` ✅ (cross-sectional, compares across stocks)
- `Rank($pb_ratio)` ✅
- `Zscore($pb_ratio)` ✅
- `Sub($pb_ratio, Mean($pb_ratio))` ✅

### Option 2: Update Fundamental Data More Frequently

Modify CSV generation to include intra-quarter estimates or use forward-fill with noise.

### Option 3: Combine Fundamental with Price Data

Create factors that combine fundamentals with price changes:
- `Mul($pb_ratio, Ret(Close, 5d))` ✅
- `Div(Ret(Close, 20d), $pe_ratio)` ✅

## Next Steps

1. **Verify**: Run new test with `unique_values` debug to confirm pb_ratio is constant
2. **Retrain**: Generate new factors using cross-sectional fundamental operators only
3. **Test**: Backtest with properly designed fundamental factors

## Related: Train/Test Split Issue (Data Leakage)

### Problem Found in [train_alphagen_from_history.py:147-154](lean_project/train_alphagen_from_history.py#L147-L154)

**Current Code (WRONG):**
```python
train_start = end_date - relativedelta(months=args.train_months)
# Test on earlier data if available  ← BUG!
test_end = train_start - timedelta(days=1)
test_start = test_end - relativedelta(months=1)
```

**Example with end_date=2023-07-31, train_months=6:**
```
train_start = 2023-07-31 - 6 months = 2023-01-31
Train: 2023-01-31 to 2023-07-31  ← FUTURE ❌

test_end = 2023-01-31 - 1 day = 2023-01-30
test_start = 2023-01-30 - 1 month = 2022-12-30
Test:  2022-12-30 to 2023-01-30  ← PAST ❌
```

**This is DATA LEAKAGE!** Model trains on future data (2023-01-31 to 2023-07-31), then tests on past data (2022-12-30 to 2023-01-30).

### Correct Implementation

```python
# Train on earlier data
train_end = end_date - relativedelta(months=1)  # Leave 1 month for testing
train_start = train_end - relativedelta(months=args.train_months)

# Test on later data (most recent)
test_start = train_end + timedelta(days=1)
test_end = end_date
```

**Example with end_date=2023-07-31, train_months=6:**
```
train_end = 2023-07-31 - 1 month = 2023-06-30
train_start = 2023-06-30 - 6 months = 2022-12-30
Train: 2022-12-30 to 2023-06-30  ← PAST ✅

test_start = 2023-06-30 + 1 day = 2023-07-01
test_end = 2023-07-31
Test:  2023-07-01 to 2023-07-31  ← FUTURE ✅
```

### Impact

- All historical training in backtests used **leaked future data**
- Test IC scores are **unreliable** (testing on past data that model already saw future of)
- Factor selection during RL training was **biased**

### Fix Required

File: [lean_project/train_alphagen_from_history.py](lean_project/train_alphagen_from_history.py#L147-154)

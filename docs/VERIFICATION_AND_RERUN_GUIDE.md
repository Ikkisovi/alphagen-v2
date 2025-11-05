# AlphaGen-Lean System Verification and Rerun Guide

Last updated: 2025-10-31

## Quick Status Check

Run the verification script to check if everything is configured correctly:

```bash
python scripts/verify_system.py
```

This script checks:
1. Directory structure
2. Fundamental data files (parquet + CSVs)
3. Factor output from training
4. Configuration files
5. Environment variables
6. Data loading capabilities

---

## Current System Status

### Files and Data
- ✅ **fundamentals.parquet**: 40,404 rows, 13 columns (0.08 MB)
- ✅ **CSV files**: 42 stock fundamental files in `manual_fundamentals/`
- ✅ **factor_output.json**: Contains 2 trained factors using fundamental data
  - Factor 1: `Mad($pb_ratio,5d)` (weight: +0.1823)
  - Factor 2: Uses `$ev_to_ebitda` and `$ps_ratio` (weight: -0.0491)
- ✅ **strategy_config.json**: 42 stocks, 6-month training window
- ✅ **main.py**: Lean strategy with ManualFundamental class

### Performance Metrics
- **Train IC**: 0.1892
- **Test IC**: 0.2308
- **Test RankIC**: 0.1815
- **Training date**: 2025-10-31 16:44

### Configuration
- **Ticker pool**: 42 stocks
- **Train window**: 6 months
- **Train steps**: 2000
- **Pool capacity**: 10
- **Forward horizon**: 20 days
- **Rebalance frequency**: MONTHLY
- **Long/Short mode**: False

---

## When to Rebuild Fundamental Data

You **ONLY need to rerun** `scripts/build_fundamental_dataset.py` if:

1. **New fundamental data**: `uniorganize_fundamental_data.txt` has been updated with new data
2. **New stocks**: You want to add new tickers to the pool
3. **Missing data**: You discovered missing or corrupted fundamental files
4. **Shares outstanding update**: You want to re-fetch shares outstanding from yfinance

Otherwise, **skip this step** and use existing data.

### To Rebuild Fundamental Data

```bash
python scripts/build_fundamental_dataset.py
```

**Expected output:**
- Creates/updates `lean_project/data/fundamentals/fundamentals.parquet`
- Creates/updates 42 CSV files in `lean_project/data/equity/usa/custom/manual_fundamentals/`
- Only warning should be for GBBK (404 from yfinance, uses static fallback)

---

## When to Retrain Factors

You **SHOULD retrain** if:

1. **New fundamental data**: After rebuilding fundamental dataset
2. **Different time period**: You want to train on a different date range
3. **Parameter changes**: Modified training parameters (steps, pool capacity, etc.)
4. **Poor IC**: Current factors have low IC scores
5. **Different stocks**: Changed the ticker pool in `strategy_config.json`

### To Retrain Factors

**Basic training (uses config from strategy_config.json):**
```bash
python lean_project/train_for_lean.py
```

**Custom training (override parameters):**
```bash
python lean_project/train_for_lean.py \
  --ticker-pool '["MU","TTMI","CDE","KGC","COMM","STRL","DXPE","WLDN","SSRM","LRN"]' \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --train-months 6 \
  --steps 2000 \
  --forward-horizon 20 \
  --output lean_project/factor_output.json
```

**Expected output:**
- Loads minute data for all tickers (may take a few minutes)
- Shows "Dataset: ... features=19" (confirming fundamentals loaded)
- PPO training progress for N steps
- Final IC scores (Train IC, Test IC, RankIC)
- Writes `lean_project/factor_output.json`

**Important environment variable:**
```bash
# Windows
set LEAN_DATA_PATH=E:\Lean\Data

# Linux/Mac
export LEAN_DATA_PATH=/path/to/lean/data
```

---

## How to Run Lean Backtest

### 1. Check Lean Configuration

Ensure your backtest dates align with your data range:

In [main.py](lean_project/main.py#L98-L99):
```python
self.SetStartDate(2023, 1, 1)   # Adjust to your data range
self.SetEndDate(2023, 12, 31)   # Adjust to your data range
```

### 2. Run Backtest

**Using Lean CLI:**
```bash
lean backtest "lean_project"
```

**Using Lean CLI with specific config:**
```bash
lean backtest "lean_project" --release
```

### 3. Check Backtest Output

Look for these key indicators in the logs:
- ✅ "Adding securities to universe..." - Stocks being added
- ✅ "[Fundamental] {ticker} {date} pe_ratio=..." - Fundamental data loading
- ✅ "INITIAL ALPHAGEN TRAINING (during warmup)" - Training triggered
- ✅ "LOADED N ALPHAGEN FACTORS" - Factors loaded successfully
- ✅ "REBALANCE #N @ {date}" - Periodic rebalancing
- ✅ Factor rankings with scores

**Expected rebalance pattern:**
- Rebalance frequency: MONTHLY (configured in strategy_config.json)
- First rebalance: After warmup (182 days)
- Subsequent rebalances: Every month on first trading day

---

## Common Issues and Solutions

### Issue 1: "LEAN_DATA_PATH not set"

**Solution:**
```bash
# Windows
set LEAN_DATA_PATH=E:\Lean\Data

# Linux/Mac
export LEAN_DATA_PATH=/path/to/lean/data
```

### Issue 2: "fundamentals.parquet not found"

**Solution:**
```bash
python scripts/build_fundamental_dataset.py
```

### Issue 3: "factor_output.json not found"

**Solution:**
```bash
python lean_project/train_for_lean.py
```

### Issue 4: Training shows "features=6" instead of "features=19"

**Problem:** Fundamentals not being loaded

**Solution:**
1. Check if fundamentals.parquet exists: `ls lean_project/data/fundamentals/`
2. Rebuild if missing: `python scripts/build_fundamental_dataset.py`
3. Verify LEAN_DATA_PATH is set correctly
4. Retrain: `python lean_project/train_for_lean.py`

### Issue 5: Backtest fails with "Symbol not found"

**Problem:** Missing minute data for some tickers

**Solution:**
1. Check which tickers are failing
2. Either remove them from ticker_pool in strategy_config.json
3. Or download minute data for those tickers via Lean CLI

### Issue 6: Low IC scores (< 0.05)

**Possible causes:**
- Insufficient training data
- Poor data quality
- Market regime change
- Need more training steps

**Solutions:**
- Increase training window: `--train-months 12`
- Increase training steps: `--steps 5000`
- Try different time periods
- Check data quality with `scripts/verify_system.py`

### Issue 7: Rebalancing too frequent/infrequent

**Solution:**
Edit `strategy_config.json`:
```json
{
  "rebalance_frequency": "MONTHLY"  // Options: "MONTHLY", "QUARTERLY", "WEEKLY"
}
```

Or modify the rebalance logic in [main.py](lean_project/main.py#L534-L548).

---

## Complete Rerun Workflow

### Scenario 1: Full System Rebuild (Start from Scratch)

```bash
# 1. Verify current system
python scripts/verify_system.py

# 2. Rebuild fundamental data
python scripts/build_fundamental_dataset.py

# 3. Train factors (set LEAN_DATA_PATH first!)
set LEAN_DATA_PATH=E:\Lean\Data  # Windows
python lean_project/train_for_lean.py

# 4. Verify again
python scripts/verify_system.py

# 5. Run backtest
lean backtest "lean_project"
```

### Scenario 2: Retrain with New Parameters

```bash
# 1. Edit strategy_config.json (or use command-line args)
# 2. Retrain
python lean_project/train_for_lean.py --steps 5000 --train-months 12

# 3. Run backtest
lean backtest "lean_project"
```

### Scenario 3: Update Fundamental Data Only

```bash
# 1. Update uniorganize_fundamental_data.txt with new data
# 2. Rebuild fundamental dataset
python scripts/build_fundamental_dataset.py

# 3. Retrain factors to use new data
python lean_project/train_for_lean.py

# 4. Run backtest
lean backtest "lean_project"
```

### Scenario 4: Quick Verification (No Changes)

```bash
# Just verify everything is configured correctly
python scripts/verify_system.py

# If all checks pass, go directly to backtest
lean backtest "lean_project"
```

---

## Testing Components

### Test Individual Components

```bash
# Test window manager, expression converter, config
python scripts/test_components.py
```

**Expected output:**
- ✅ Window Manager Test PASSED
- ✅ Expression Converter Test PASSED (9/9 passed)
- ✅ Batch Conversion Test PASSED
- ✅ Configuration Test PASSED

### Test Data Loading

```python
# In Python REPL
import pandas as pd
df = pd.read_parquet('lean_project/data/fundamentals/fundamentals.parquet')
print(df.shape)  # Should be (40404, 13)
print(df.columns.tolist())  # Should include fundamental columns
```

---

## Monitoring Backtest Progress

### Real-time Log Monitoring

**Windows:**
```bash
Get-Content lean_project\backtests\*.log -Wait
```

**Linux/Mac:**
```bash
tail -f lean_project/backtests/*.log
```

### Key Log Messages to Watch For

1. **Initialization:**
   - "Adaptive AlphaGen Live Strategy Initialized"
   - "Ticker pool size: 42"

2. **Warmup Training:**
   - "INITIAL ALPHAGEN TRAINING (during warmup)"
   - "AlphaGen Training Output"
   - "Dataset: ... features=19"

3. **Factor Loading:**
   - "LOADED N ALPHAGEN FACTORS"
   - "Train IC: ... | Test IC: ..."

4. **Rebalancing:**
   - "REBALANCE #N @ {date}"
   - "Factor Rankings (Top 10)"
   - "Target Portfolio: N positions"

5. **Summary:**
   - "ALGORITHM SUMMARY"
   - "Total Return: X.XX%"

---

## Performance Expectations

### Training Time
- **2000 steps**: ~5-15 minutes (depending on CPU)
- **5000 steps**: ~15-45 minutes

### Backtest Time
- **1 year, 42 stocks**: ~2-10 minutes
- Initial training during warmup takes most of the time

### IC Score Expectations
- **Good**: IC > 0.15, RankIC > 0.10
- **Acceptable**: IC > 0.08, RankIC > 0.05
- **Poor**: IC < 0.05 (consider retraining with different parameters)

### Current Performance
- **Train IC**: 0.189 ✅ (Good)
- **Test IC**: 0.231 ✅ (Good)
- **Test RankIC**: 0.181 ✅ (Good)

---

## Next Steps

Now that everything is verified and working:

1. **Run a backtest** to see how the strategy performs
2. **Monitor the logs** for any errors or warnings
3. **Analyze results** in Lean's backtest report
4. **Iterate** by adjusting parameters or adding more features
5. **Consider live trading** once backtest results are satisfactory

---

## Getting Help

If you encounter issues:

1. Run `python scripts/verify_system.py` to diagnose
2. Check the "Common Issues" section above
3. Review backtest logs for specific error messages
4. Verify all file paths and environment variables
5. Ensure Lean data directory has minute data for all tickers

---

## Summary

Your system is **currently working correctly** based on verification:

- ✅ All fundamental data files present
- ✅ Factor training completed successfully
- ✅ Configuration files valid
- ✅ Data loading works
- ✅ Component tests pass

**You do NOT need to rerun anything unless:**
- You want to update data
- You want to retrain with different parameters
- You want to test different time periods

**To run a backtest right now:**
```bash
lean backtest "lean_project"
```

That's it! The system is ready to use.

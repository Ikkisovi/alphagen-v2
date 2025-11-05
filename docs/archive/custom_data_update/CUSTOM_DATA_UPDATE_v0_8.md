# Custom Data Integration v0.8 - Minute Data Symlinks

**Date**: 2025-11-01
**Previous**: v0.7 - Lean core files
**Status**: Market data linked, ready for backtest

## Issue Found

**Problem**: 100% failed data requests, "insufficient data" for all stocks
**Root cause**: Minute/daily data files exist in `/e/factor/lean_project/data/` but not in `/e/factor/alphagen/lean_project/data/`

### Why This Happened
- Custom fundamental CSV files created in `alphagen/lean_project/data/`
- But equity minute/daily data still only in `lean_project/data/`
- Lean couldn't find price data → daily_bars never filled → "insufficient data"

## Fix Applied

Created symlinks to share market data between projects:

```bash
cd /e/factor/alphagen/lean_project/data/equity/usa
ln -s /e/factor/lean_project/data/equity/usa/minute minute
ln -s /e/factor/lean_project/data/equity/usa/daily daily
ln -s /e/factor/lean_project/data/equity/usa/map_files map_files
ln -s /e/factor/lean_project/data/equity/usa/factor_files factor_files
```

## Directory Structure Now

```
alphagen/lean_project/data/
├── equity/usa/
│   ├── custom/manual_fundamentals/  # Real files - 42 CSV
│   ├── minute/  → symlink to /e/factor/lean_project/data/equity/usa/minute
│   ├── daily/   → symlink
│   ├── map_files/   → symlink
│   └── factor_files/ → symlink
├── fundamentals/fundamentals.parquet
├── market-hours/
└── symbol-properties/
```

## Benefits of Symlinks
- No data duplication (minute data is large)
- Single source of truth for market data
- Custom fundamentals stay separate
- Easy to update market data in one place

## Expected Behavior After Fix

**Data loading:**
- Lean finds minute data for all tickers
- OnData aggregates minute → daily bars
- daily_bars fills up during warmup (182 days)
- Fundamental data merges with price data

**First rebalance:**
- Should have ~180 days of data per stock
- Factor calculation succeeds
- Non-zero differentiated factor values
- Stocks ranked and selected based on fundamentals

## Files Changed

1. Created symlinks in `lean_project/data/equity/usa/`:
   - `minute/` → original data
   - `daily/` → original data
   - `map_files/` → original data
   - `factor_files/` → original data

## Verification

Run backtest:
```bash
lean backtest "lean_project"
# Select: 1) e:\factor\alphagen\lean_project\lean.json
```

Should now:
1. Load minute data successfully
2. Aggregate to daily bars
3. Merge with fundamentals
4. Calculate factors with real values
5. Select stocks based on pb_ratio and ev_to_ebitda

## System Status v0.8

- ✅ Fundamental CSV files (42 stocks)
- ✅ Factor output (2 factors using fundamentals)
- ✅ Import fixed
- ✅ Data path fixed
- ✅ Lean core files added
- ✅ Minute/daily data linked
- ✅ **Ready for successful backtest**

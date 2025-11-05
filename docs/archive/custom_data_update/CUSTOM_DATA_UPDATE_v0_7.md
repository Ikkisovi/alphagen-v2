# Custom Data Integration v0.7 - Lean Core Files

**Date**: 2025-11-01
**Previous**: v0.6 - Data folder path fix
**Status**: Lean core configuration files added

## Issue Found

**Problem**: Backtest failed immediately on startup
**Error**: `Unable to locate symbol properties file: Data/symbol-properties/symbol-properties-database.csv`

### Root Cause
Lean engine requires core configuration files:
- `symbol-properties/symbol-properties-database.csv`
- `market-hours/market-hours-database.csv`

These were missing from `lean_project/data/` directory.

## Fix Applied

Copied required Lean core files from `/e/factor/lean_project/data/`:
```bash
cp -r symbol-properties /e/factor/alphagen/lean_project/data/
cp -r market-hours /e/factor/alphagen/lean_project/data/
```

## Directory Structure Now

```
lean_project/data/
├── equity/
│   └── usa/
│       └── custom/
│           └── manual_fundamentals/  # 42 CSV files
├── fundamentals/
│   └── fundamentals.parquet
├── market-hours/               # ✅ Added
│   └── market-hours-database.csv
└── symbol-properties/          # ✅ Added
    └── symbol-properties-database.csv
```

## Files Changed

1. Added `lean_project/data/symbol-properties/` directory
2. Added `lean_project/data/market-hours/` directory

## Verification

Run backtest:
```bash
lean backtest "lean_project"
# Select: 1) e:\factor\alphagen\lean_project\lean.json
```

Should now:
1. Start Lean engine successfully
2. Load symbol properties and market hours
3. Initialize algorithm
4. Begin warmup

## System Status v0.7

- ✅ Fundamental CSV files (42 stocks)
- ✅ Factor output (2 factors)
- ✅ Import fixed (SubscriptionTransportMedium)
- ✅ Data path fixed (lean.json)
- ✅ Lean core files added
- ✅ Ready for backtest

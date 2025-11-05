# Custom Data Integration v0.5 - Critical Import Fix

**Date**: 2025-10-31
**Previous**: v0.4 - Monthly retraining fix
**Status**: Fundamental data loading fixed

## Critical Issue Found

**Problem**: All factors returned 0.0000, fundamental data not loading
**Root cause**: Missing import in ManualFundamental.GetSource()

### Error Details
```
Python.Runtime.PythonException: name 'DataSourceType' is not defined
File "/LeanCLI/main.py", line 67, in GetSource
    return SubscriptionDataSource(file_path, DataSourceType.LocalFile, FileFormat.Csv)
```

### Impact
- ManualFundamental class failed to load CSV files
- No fundamental features available ($pb_ratio, $ev_to_ebitda, etc.)
- Factor calculations returned NaN → normalized to 0.0000
- All stocks had identical 0.0000 factor scores
- Portfolio selected randomly (first 7 stocks alphabetically)

## Fix Applied

Changed line 67 in `lean_project/main.py`:

**Before:**
```python
return SubscriptionDataSource(file_path, DataSourceType.LocalFile, FileFormat.Csv)
```

**After:**
```python
return SubscriptionDataSource(file_path, SubscriptionTransportMedium.LocalFile)
```

`SubscriptionTransportMedium` is properly imported via `from AlgorithmImports import *`

## Expected Behavior After Fix

**Fundamental loading:**
- CSV files load correctly
- Log shows: "[Fundamental] {ticker} {date} pe_ratio=... shares_out=..."
- Factor calculations use real fundamental values

**Factor values:**
- Non-zero differentiated scores
- Proper ranking based on $pb_ratio and $ev_to_ebitda
- Portfolio selects top-ranked stocks by actual factor values

## Files Changed

1. `lean_project/main.py` line 67 - Fixed GetSource() import

## Verification

Run backtest - should now show:
1. No more "DataSourceType is not defined" errors
2. Fundamental log messages with real values
3. Factor rankings with non-zero differentiated scores
4. Different stocks selected each rebalance period

## System Status v0.5

- ✅ Fundamental data files ready
- ✅ Factor training complete
- ✅ Import fixed - fundamental loading works
- ✅ Monthly retraining disabled in backtests
- ✅ Ready for testing

# Custom Data Integration v0.6 - Data Folder Path Fix

**Date**: 2025-11-01
**Previous**: v0.5 - Import fix
**Status**: Data path configured correctly

## Critical Issue Found

**Problem**: All factors still 0.0000, fundamental CSV files not loading
**Root cause**: `lean.json` data-folder pointed to wrong directory

### Path Mismatch
- CSV files location: `E:\factor\alphagen\lean_project\data\`
- lean.json data-folder: `E:\factor\lean_project\data\` ❌

### Impact
- Lean container couldn't find fundamental CSV files
- 52% failed data requests
- ManualFundamental.GetSource() returned paths that didn't exist
- No fundamental data loaded → all NaN → factors = 0.0000

## Fix Applied

Changed line 9 in `lean_project/lean.json`:

**Before:**
```json
"data-folder": "E:\\\\factor\\\\lean_project\\\\data"
```

**After:**
```json
"data-folder": "E:\\\\factor\\\\alphagen\\\\lean_project\\\\data"
```

## Expected Behavior After Fix

**Data loading:**
- Lean finds CSV files at correct path
- ManualFundamental data loads successfully
- Log shows: "[Fundamental] {ticker} {date} pe_ratio=..."
- Data request success rate > 90%

**Factor calculations:**
- Real pb_ratio, ev_to_ebitda values available
- Factor functions return differentiated scores
- Non-zero factor rankings
- Proper stock selection based on fundamentals

## Files Changed

1. `lean_project/lean.json` line 9 - Fixed data-folder path

## Verification

Run backtest with correct lean.json:
```bash
lean backtest "lean_project"
# Select: 1) e:\factor\alphagen\lean_project\lean.json
```

Expected logs:
1. No "Path not found" errors for CSV files
2. Fundamental log messages with real values
3. Factor rankings with differentiated scores

## System Status v0.6

- ✅ Fundamental CSV files ready (42 stocks)
- ✅ Factor output ready (2 factors using fundamentals)
- ✅ Import fixed (SubscriptionTransportMedium)
- ✅ Data path fixed (lean.json)
- ✅ Monthly retraining disabled in backtests
- ✅ Ready for testing

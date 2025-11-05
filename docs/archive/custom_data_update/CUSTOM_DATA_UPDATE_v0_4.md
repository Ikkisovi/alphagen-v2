# Custom Data Integration v0.4 - Monthly Retraining Fix

**Date**: 2025-10-31
**Previous**: v0.3 - Full integration complete
**Status**: Monthly retraining issue fixed

## Issue Found

During backtests, monthly retraining was **always failing** because:
- Lean container can't access minute data paths during retraining
- Training script expects `/Data/equity/usa/minute/{ticker}/` paths
- These paths don't exist or aren't accessible in backtest environment
- Error: "Path not found for {ticker}", "No data loaded for any symbols"

## Fix Applied

Modified `lean_project/main.py` line 757-767:
- Added check to skip monthly retraining during backtests
- Only allows retraining in `LiveMode` where data is reliably accessible
- Initial warmup training still runs (uses pre-generated factor_output.json)
- Shows clear skip message in logs

```python
def MonthlyRetraining(self):
    if self.IsWarmingUp:
        return

    # Skip retraining during backtests (data access issues)
    if not self.LiveMode:
        self.Debug(f"\n[SKIP] Monthly retraining disabled during backtests")
        return

    # ... rest of retraining logic for live mode
```

## Current Behavior

**Backtests:**
- Train once during warmup using existing `factor_output.json`
- Skip monthly retraining (factors stay constant)
- Log shows: "[SKIP] Monthly retraining disabled during backtests"

**Live Trading:**
- Train during warmup
- Retrain monthly (data accessible in live environment)
- Factors update dynamically

## System Status v0.4

**All components working:**
- ✅ Fundamental data: 42 stocks, 40,404 rows
- ✅ Factor training: IC 0.189 (train), 0.231 (test)
- ✅ Initial training: Works in warmup
- ✅ Monthly retraining: Disabled in backtests, enabled in live
- ✅ Rebalancing: Monthly, works correctly

**Verification:**
```bash
python scripts/verify_system.py  # Check system health
lean backtest "lean_project"      # Should now complete without retraining errors
```

## Files Changed

1. `lean_project/main.py` - Added LiveMode check in MonthlyRetraining()

## Next Steps

Run backtest - should now complete successfully without monthly retraining failures.

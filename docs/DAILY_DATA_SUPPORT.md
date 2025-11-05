# Daily Data Support Update

## Overview

This update adds support for directly using daily resolution data downloaded from Lean API, eliminating the need for minute-to-daily aggregation.

## Changes Made

### 1. Data Preparation Module ([alphagen_lean/data_prep.py](lean_project/alphagen_lean/data_prep.py))

**New Features:**
- Added `resolution` parameter to `LeanDataLoader` class (supports "minute" or "daily")
- Implemented `load_daily_lean_data()` method to load Lean daily format data
- Updated `load_all_symbols_data()` to route to appropriate loader based on resolution
- Updated `prepare_window_data()` to accept and pass through resolution parameter

**Daily Data Format:**
```
- Stored in: {LEAN_DATA_PATH}/equity/usa/daily/{symbol}/{symbol}.zip
- Contains: {symbol}.csv
- Columns: [date, open, high, low, close, volume]
- Date format: YYYYMMDD (integer)
- Prices: Scaled by 10000 (automatically divided)
```

### 2. Local Data Module ([alphagen/local_data.py](alphagen/local_data.py))

**Smart Aggregation:**
- Modified `_aggregate_to_daily()` to auto-detect data resolution
- If data is already daily (one row per date/symbol), skip aggregation
- If data is intraday (multiple rows per date/symbol), aggregate as before
- Added informative console output about detected resolution

**Detection Logic:**
```python
# Check if each (date, symbol) pair has exactly one row
grouped_counts = df.groupby(["date", "symbol"]).size()
is_daily = (grouped_counts == 1).all()
```

### 3. Training Script ([lean_project/train_for_lean.py](lean_project/train_for_lean.py))

**New CLI Parameter:**
```bash
--resolution {minute,daily}  # Default: daily
```

**Path Resolution:**
- `--resolution daily` → Uses `/Data/equity/usa/daily/`
- `--resolution minute` → Uses `/Data/equity/usa/minute/`

## Usage

### Using Daily Data (Recommended)

```bash
python lean_project/train_for_lean.py \
  --ticker-pool '["MU","TTMI","CDE"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution daily \
  --output output/factors.json
```

### Using Minute Data (Legacy)

```bash
python lean_project/train_for_lean.py \
  --ticker-pool '["MU","TTMI","CDE"]' \
  --start-date 2023-11-29 \
  --end-date 2024-06-28 \
  --resolution minute \
  --output output/factors.json
```

## Testing

A test script is provided to verify the daily data loading:

```bash
python lean_project/test_daily_data.py
```

This script will:
1. Load daily data for sample symbols
2. Verify data structure and resolution
3. Display data summary and samples

## Benefits

1. **Performance**: No minute-to-daily aggregation overhead
2. **Simplicity**: Direct API subscription matches training data
3. **Consistency**: Same resolution for training and backtesting
4. **Backward Compatible**: Still supports minute data aggregation

## Data Download

To download daily data using Lean CLI:

```bash
cd lean_project
lean data download --dataset "US Equity Security Master" --resolution daily
```

For specific symbols:
```bash
lean data download --dataset "US Equity Security Master" \
  --resolution daily \
  --ticker MU TTMI CDE KGC COMM
```

## Architecture Notes

### Data Flow (Daily Resolution)

```
Lean Daily Data (.zip files)
    ↓
LeanDataLoader.load_daily_lean_data()
    ↓
DataFrame with daily OHLCV
    ↓
_aggregate_to_daily() [detects daily, skips aggregation]
    ↓
StockData tensor for AlphaGen
```

### Data Flow (Minute Resolution - Legacy)

```
Lean Minute Data (.zip files)
    ↓
LeanDataLoader.load_real_lean_data()
    ↓
DataFrame with minute OHLCV
    ↓
_aggregate_to_daily() [detects intraday, aggregates]
    ↓
StockData tensor for AlphaGen
```

## Files Modified

1. [lean_project/alphagen_lean/data_prep.py](lean_project/alphagen_lean/data_prep.py)
   - Added resolution support
   - Implemented daily data loader

2. [alphagen/local_data.py](alphagen/local_data.py)
   - Smart aggregation detection
   - Skips aggregation for daily data

3. [lean_project/alphagen/local_data.py](lean_project/alphagen/local_data.py)
   - Synchronized with main local_data.py

4. [lean_project/train_for_lean.py](lean_project/train_for_lean.py)
   - Added --resolution parameter
   - Automatic path resolution

5. [lean_project/test_daily_data.py](lean_project/test_daily_data.py) (New)
   - Verification and testing script

## Next Steps

1. **Verify Daily Data Download**
   ```bash
   # Check if daily data exists
   ls -la /Data/equity/usa/daily/
   ```

2. **Run Test Script**
   ```bash
   python lean_project/test_daily_data.py
   ```

3. **Train with Daily Data**
   ```bash
   # Use your existing ticker pool
   python lean_project/train_for_lean.py \
     --ticker-pool "$(cat lean_project/data/ticker_pool.json)" \
     --start-date 2023-11-29 \
     --end-date 2024-06-28 \
     --resolution daily \
     --steps 6000 \
     --output lean_project/output/factors_daily.json
   ```

4. **Compare Results**
   - Train with both resolutions
   - Compare IC scores and factor expressions
   - Verify backtest performance

## Troubleshooting

### Issue: "Daily data zip not found"
**Solution**: Download daily data using `lean data download --resolution daily`

### Issue: "Data is still being aggregated"
**Cause**: Data has multiple rows per (date, symbol)
**Solution**: Check data source - should be daily resolution

### Issue: Missing VWAP column
**Solution**: VWAP is approximated using close price for daily data

## Performance Impact

Expected improvements with daily data:
- **Data Loading**: ~10x faster (no intraday bars to load)
- **Aggregation**: Skipped (instant)
- **Memory**: ~100x less (daily vs minute granularity)
- **Training**: Same (operates on daily tensors regardless)

## Compatibility

- ✅ Backward compatible with existing minute data workflows
- ✅ Works with two-stage training (price-only + fundamentals)
- ✅ Compatible with warm-start functionality
- ✅ Works with Lean backtest integration

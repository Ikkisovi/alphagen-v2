# AlphaGen Ensemble Training - Test Run Guide

## Overview

This guide helps you prepare and run the ensemble training system with merged OHLCV and fundamental data.

## Data Preparation Status

✅ **Completed Steps:**

1. **Fundamental Data with SharesOut**
**shares_outstanding**

2. **OHLCV Data**
   - Location: `e:/factor/lean_project/data/equity/usa/daily/`
   - Format: ZIP files containing CSV data
   - Coverage: 45 symbols from 2014-04-03 to 2025-10-31

3. **Merged Dataset**
   - Output: `data/merged/merged_data.parquet`
   - Combined: OHLCV + Fundamentals
   - Size: 33,604 rows covering 45 symbols
   - Date range: 2022-01-04 to 2025-10-31
   - Script: `scripts/merge_ohlcv_fundamentals.py`

## System Verification

Run system verification to check all components:

```bash
python scripts/verify_system.py
```

**Expected Output:**
- ✓ Python Version: 3.10.11
- ✓ Dependencies: All required packages installed
- ✓ Data Files: Merged and fundamental data available

## Ensemble Training Configuration

The ensemble training uses a dual-stage approach:

### Stage 1: Technical/Price-Only Sweep
- Uses only OHLCV data
- Discovers ~80 candidate factors
- 500 episodes with early stopping

### Stage 2: Fundamentals Enhancement
- Takes top 25% from Stage 1
- Adds fundamental features (PE, PB, PS, etc.)
- Refines to ~20 factors per window

### Multi-Window Training
- **12m window**: 2023-01-01 to 2023-12-31
- **6m window**: 2023-07-01 to 2023-12-31
- **3m window**: 2023-10-01 to 2023-12-31
- **Validation**: 2024-01-01 to 2024-03-31

## Running the Test

### Quick Test (Recommended for First Run)

Use the test script with reduced episodes:

```bash
python scripts/test_ensemble_training.py
```

This will:
- Verify data availability
- Run with reduced episodes (50 instead of 500)
- Create test outputs in `output/test_ensemble/`
- Validate all components work correctly

**Expected Runtime:** 30-60 minutes (depending on hardware)

### Full Production Training

For full training with complete episodes:

```bash
python scripts/train_ensemble.py --config config/ensemble_config.yaml
```

**Expected Runtime:** 4-8 hours (depending on hardware and GPU availability)

### Using Cached Results

If you've already trained and want to reuse pools:

```bash
python scripts/train_ensemble.py --config config/ensemble_config.yaml --skip-training
```

## Output Structure

```
output/
├── ensemble/                    # Production outputs
│   ├── pool_12m.json           # 12-month window factors
│   ├── pool_6m.json            # 6-month window factors
│   ├── pool_3m.json            # 3-month window factors
│   ├── ensemble_pool_2024Q1.json  # Final merged ensemble
│   ├── ensemble_training.log   # Training logs
│   ├── ensemble_report_2024Q1.txt # Performance report
│   └── ensemble_metrics_2024Q1.json # Detailed metrics
│
└── test_ensemble/              # Test run outputs
    └── [same structure as above]
```

## Key Scripts

1. **`scripts/build_fundamental_dataset.py`**
   - Parses fundamental txt file
   - Fetches shares outstanding via yfinance
   - Exports parquet and CSV files

2. **`scripts/merge_ohlcv_fundamentals.py`**
   - Merges OHLCV with fundamentals
   - Aligns dates and symbols
   - Exports combined dataset

3. **`scripts/train_ensemble.py`**
   - Main ensemble training orchestrator
   - Dual-stage training per window
   - Cross-validated optimization

4. **`scripts/test_ensemble_training.py`**
   - Quick validation test
   - Reduced episodes for speed
   - Verifies pipeline end-to-end

5. **`scripts/verify_system.py`**
   - System health check
   - Dependency verification
   - Data availability check

## Troubleshooting

### Issue: "qlib not initialized"

```bash
# Initialize qlib with your data directory
qlib init --qlib_dir ~/.qlib
```

### Issue: "CUDA out of memory"

Edit `config/ensemble_config.yaml`:
```yaml
training:
  device: "cpu"  # Use CPU instead of GPU
```

### Issue: "No module named 'alphagen'"

Make sure you're in the project root directory:
```bash
cd e:/factor/alphagen
python scripts/verify_system.py
```

### Issue: Missing data files

Re-run data preparation:
```bash
python scripts/build_fundamental_dataset.py
python scripts/merge_ohlcv_fundamentals.py
```

## Next Steps After Successful Test

1. **Analyze Results**
   - Review `output/test_ensemble/test_report.txt`
   - Check factor quality and IC values
   - Examine top weighted factors

2. **Run Full Training**
   - Use full configuration: `python scripts/train_ensemble.py`
   - Monitor logs: `tail -f output/ensemble/ensemble_training.log`

3. **Deploy to Lean**
   - Use factors from `ensemble_pool_2024Q1.json`
   - Integrate with Lean algorithm
   - Backtest with full universe

## Performance Expectations

- **Factor Discovery**: 60-80 factors per window
- **Final Ensemble**: 30-45 factors after optimization
- **Validation IC**: Target > 0.03 (3% information coefficient)
- **Weight Sparsity**: L1 regularization promotes sparse weights

## Data Quality Notes

Current merged dataset shows:
- **0%** nulls in OHLCV data (complete)
- **1.64%** nulls in most fundamental ratios (acceptable)
- **19.54%** nulls in EV to EBITDA (some companies don't report)
- **0%** nulls in shares_outstanding (successfully fetched)

## Support

For issues or questions:
1. Check logs in `output/ensemble/ensemble_training.log`
2. Run `python scripts/verify_system.py` for diagnostics
3. Review configuration in `config/ensemble_config.yaml`

---

**Status:** Ready for test run ✓

All data prepared, system verified, scripts ready. Run `python scripts/test_ensemble_training.py` to begin.

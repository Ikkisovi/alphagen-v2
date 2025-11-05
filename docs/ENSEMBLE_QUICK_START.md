# Ensemble Alpha Mining - Quick Start Guide

## 1-Minute Setup

```bash
# Step 1: Run tests
python scripts/test_ensemble.py

# Step 2: Run training
python scripts/train_ensemble.py

# Step 3: Check results
cat output/ensemble/ensemble_report_2024Q1.txt
```

## What It Does

Trains alpha factors on 3 time windows (12M, 6M, 3M), merges them into a robust ensemble optimized on Q1 2024 validation data.

## Key Files

| File | Purpose |
|------|---------|
| `config/ensemble_config.yaml` | Configuration |
| `scripts/train_ensemble.py` | Main training script |
| `scripts/test_ensemble.py` | Test suite |
| `output/ensemble/ensemble_pool_2024Q1.json` | Final ensemble output |

## Common Commands

**Full training**:
```bash
python scripts/train_ensemble.py
```

**Use cached pools**:
```bash
python scripts/train_ensemble.py --skip_training
```

**Custom config**:
```bash
python scripts/train_ensemble.py --config_file my_config.yaml
```

**Set random seed**:
```bash
python scripts/train_ensemble.py --seed 42
```

## Training Flow

```
12M Window (2023-01-01 → 2023-12-31)  →  ~20 factors
6M Window  (2023-07-01 → 2023-12-31)  →  ~20 factors
3M Window  (2023-10-01 → 2023-12-31)  →  ~20 factors
                                         ───────────
                                         ~60 factors
                                              ↓
                                    Merge + Deduplicate
                                              ↓
                                    Optimize on Q1 2024
                                              ↓
                                      ≤45 final factors
```

## Each Window Training

**Stage 1**: Technical/price features only → ~80 candidates
**Stage 2**: Top 25% + fundamentals → ~20 final factors

## Configuration Highlights

### Time Windows
```yaml
train_12m: 2023-01-01 to 2023-12-31
train_6m:  2023-07-01 to 2023-12-31
train_3m:  2023-10-01 to 2023-12-31
validation: 2024-01-01 to 2024-03-31  # Q1 2024
```

### Training
```yaml
max_episodes: 500 per window
early_stopping_patience: 30 episodes
pool_capacity: 25 per window
final_ensemble_capacity: 45
```

### Features
- **Technical**: OHLCV, VWAP, Turnover (7 features)
- **Fundamental**: Valuation, Yields, Growth, Leverage, Profitability (25 features)
- **Total**: 32 features available

### Cache
```yaml
enabled: true
freshness_days: 30  # Reuse if < 30 days old
```

## Output Structure

```
output/ensemble/
├── pool_12m.json              # 12M window (~20 factors)
├── pool_6m.json               # 6M window (~20 factors)
├── pool_3m.json               # 3M window (~20 factors)
├── ensemble_pool_2024Q1.json  # Final ensemble (≤45 factors) ⭐
├── ensemble_training.log      # Detailed log
├── ensemble_report_2024Q1.txt # Summary report
└── ensemble_metrics_2024Q1.json # Metrics
```

## Interpreting Results

### Good Results
- Validation IC > 0.04
- 35-45 factors in final ensemble
- All windows contribute >50% of factors
- Weight distribution: mean ~0, std ~0.1-0.2

### Red Flags
- Validation IC < 0.02 → Increase regularization
- <20 factors in ensemble → Reduce IC threshold
- One window dominates (>80%) → Adjust window balance

## Customization

### Change time periods
Edit `config/ensemble_config.yaml`:
```yaml
time_windows:
  train_12m:
    start_date: "2022-01-01"  # Your start
    end_date: "2022-12-31"     # Your end
```

### Add/remove features
Edit `config/ensemble_config.yaml`:
```yaml
fundamental_features:
  - "PERatio"
  - "PBRatio"
  # Add your features
```

### Adjust capacity
```yaml
ensemble:
  final_capacity: 60  # More factors
```

### Change regularization
```yaml
ensemble:
  optimizer:
    l1_alpha: 0.01  # Stronger regularization (fewer factors)
```

## Troubleshooting

### Out of memory
```yaml
training:
  device: "cpu"  # Use CPU instead of GPU
  pool_capacity_per_window: 15  # Reduce capacity
```

### Training too slow
```yaml
training:
  stage1_technical:
    max_episodes: 200  # Reduce episodes
  stage2_fundamentals:
    max_episodes: 200
```

### Cache not working
```bash
# Check cache directory
ls -la output/cache/

# Force fresh training
rm output/ensemble/pool_*.json
python scripts/train_ensemble.py
```

## Next Steps

1. **Understand the architecture**: Read [ENSEMBLE_TRAINING.md](ENSEMBLE_TRAINING.md)
2. **Customize config**: Edit [config/ensemble_config.yaml](../config/ensemble_config.yaml)
3. **Run experiments**: Try different time windows, features, hyperparameters
4. **Deploy to Lean**: Use the final ensemble in your trading algorithm

## Example Session

```bash
# Terminal 1: Run training
python scripts/train_ensemble.py

# Terminal 2: Monitor progress (if using tensorboard)
tensorboard --logdir output/ensemble/

# After training completes
cat output/ensemble/ensemble_report_2024Q1.txt

# Sample output:
# ================================================================================
# ENSEMBLE SUMMARY
# ================================================================================
# Total factors collected: 60
# Final ensemble size: 42
# Validation IC: 0.0538
# ================================================================================
```

## Key Metrics

- **Pool Size**: Number of factors in final ensemble (target: 35-45)
- **Validation IC**: Information Coefficient on Q1 2024 (target: >0.04)
- **Window Contribution**: % of factors from each window in final ensemble
- **Weight Distribution**: Mean ~0, most weights between -0.3 and +0.3

## Performance

Typical runtime (GPU):
- 12M window: ~2 hours
- 6M window: ~1.5 hours
- 3M window: ~1 hour
- Ensemble merge: ~5 minutes
- **Total**: ~5 hours

With cache (subsequent runs): ~10 minutes

---

**Full Documentation**: [ENSEMBLE_TRAINING.md](ENSEMBLE_TRAINING.md)
**Configuration Reference**: [config/ensemble_config.yaml](../config/ensemble_config.yaml)
**Test Suite**: `python scripts/test_ensemble.py`

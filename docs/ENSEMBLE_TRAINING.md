# Ensemble Alpha Mining - User Guide

## Overview

The Ensemble Alpha Mining system optimizes alpha factor discovery by training on multiple time windows and merging the results into a robust ensemble. This approach:

- **Captures multiple market regimes** (12M, 6M, 3M windows)
- **Uses dual-stage training** (technical → technical+fundamentals)
- **Automatically deduplicates and optimizes** factor combinations
- **Reduces overfitting** through cross-validated shrinkage
- **Supports intelligent caching** to avoid redundant computation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENSEMBLE TRAINING FLOW                    │
└─────────────────────────────────────────────────────────────┘

 STAGE 1: Window-Specific Training
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │  12M Window  │  │   6M Window  │  │   3M Window  │
 │  2023-01-01  │  │  2023-07-01  │  │  2023-10-01  │
 │  2023-12-31  │  │  2023-12-31  │  │  2023-12-31  │
 └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
        │                 │                 │
        │    Dual-Stage Training (each):    │
        │    1. Technical-only sweep        │
        │    2. Top 25% + Fundamentals      │
        │                 │                 │
        ▼                 ▼                 ▼
  ┌─────────┐       ┌─────────┐       ┌─────────┐
  │ ~20     │       │ ~20     │       │ ~20     │
  │ Factors │       │ Factors │       │ Factors │
  └────┬────┘       └────┬────┘       └────┬────┘
       └─────────────┬────────────────┘
                     ▼
 STAGE 2: Ensemble Merging
       ┌──────────────────────┐
       │   Merge ~60 Factors  │
       │   Deduplicate        │
       │   Cap at 45          │
       └──────────┬───────────┘
                  ▼
 STAGE 3: Final Optimization
       ┌──────────────────────┐
       │ Optimize on Q1 2024  │
       │ Cross-Validation     │
       │ Shrinkage (0.9)      │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │  Final Ensemble      │
       │  ≤45 Factors         │
       │  Optimized Weights   │
       └──────────────────────┘
```

## Quick Start

### 1. Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### 2. Configuration

Edit [config/ensemble_config.yaml](../config/ensemble_config.yaml) to customize:

- **Time windows**: Training and validation periods
- **Universe**: Stock selection (base + tech stocks)
- **Training params**: Episodes, early stopping, pool capacity
- **Fundamental features**: Which metrics to include
- **Ensemble settings**: Final capacity, IC thresholds
- **Cache settings**: Freshness, reuse policy

### 3. Run Tests

Validate the system before training:

```bash
python scripts/test_ensemble.py
```

Expected output:
```
================================================================================
ALL TESTS PASSED ✅
================================================================================
The ensemble training system is ready to use!
Run: python scripts/train_ensemble.py
================================================================================
```

### 4. Run Training

**Full training** (all 3 windows):

```bash
python scripts/train_ensemble.py
```

**With custom config**:

```bash
python scripts/train_ensemble.py --config_file path/to/config.yaml
```

**Skip training, use cached pools**:

```bash
python scripts/train_ensemble.py --skip_training
```

**Set random seed**:

```bash
python scripts/train_ensemble.py --seed 123
```

## Configuration Reference

### Time Windows

```yaml
time_windows:
  train_12m:
    name: "12m"
    start_date: "2023-01-01"
    end_date: "2023-12-31"
```

- **12M**: Long-term trends, stable factors
- **6M**: Medium-term momentum, seasonal effects
- **3M**: Short-term dynamics, recent market regime

### Universe

```yaml
universe:
  base_instrument: "csi300"  # or "csi500"
  additional_stocks:
    - "AAPL"   # Apple
    - "GOOGL"  # Alphabet
    - "AMZN"   # Amazon
    - "NVDA"   # NVIDIA
    - "SNDK"   # SanDisk
```

### Training Configuration

```yaml
training:
  stage1_technical:
    max_episodes: 500
    early_stopping_patience: 30
    target_candidates: 80

  stage2_fundamentals:
    max_episodes: 500
    top_quartile_only: true
    target_factors_per_window: 20
```

**Stage 1**: Fast technical sweep to find ~80 candidates
**Stage 2**: Refine top 25% with fundamentals to ~20 final

### Fundamental Features

Extended feature set (33 total):

**Valuation**: PERatio, PBRatio, PSRatio, EVToEBITDA, EVToRevenue, EVToFCF
**Yields**: EarningYield, FCFYield, SalesYield, DividendYield
**Growth**: RevenueGrowth, EarningsGrowth, BookValueGrowth
**Leverage**: DebtToAssets, DebtToEquity, CurrentRatio, QuickRatio
**Profitability**: ROE, ROA, ROIC, GrossMargin, OperatingMargin, NetMargin

### Ensemble Settings

```yaml
ensemble:
  final_capacity: 45
  optimizer:
    l1_alpha: 0.005        # L1 regularization
  cross_validation:
    enabled: true
    n_folds: 5
    shrinkage_factor: 0.9  # Reduce overfitting
```

### Cache Management

```yaml
cache:
  enabled: true
  freshness_days: 30              # Reuse if < 30 days old
  check_universe_consistency: true
  skip_training_if_cached: false  # Override to always use cache
```

## Output Files

After training, you'll find:

```
output/ensemble/
├── pool_12m.json              # 12M window pool (~20 factors)
├── pool_6m.json               # 6M window pool (~20 factors)
├── pool_3m.json               # 3M window pool (~20 factors)
├── ensemble_pool_2024Q1.json  # Final ensemble (≤45 factors)
├── ensemble_training.log      # Detailed training log
├── ensemble_report_2024Q1.txt # Human-readable report
└── ensemble_metrics_2024Q1.json # Detailed metrics (JSON)
```

### Final Ensemble Format

```json
{
  "exprs": [
    "$close",
    "Div($close, $volume)",
    "Greater(Sub($high, $low), $vwap)"
  ],
  "weights": [
    0.234,
    -0.156,
    0.089
  ]
}
```

## Performance Report

The report includes:

1. **Window Summaries**: Factors and IC per training window
2. **Ensemble Summary**: Total factors, final size, validation IC
3. **Window Contributions**: How many factors from each window made it to final ensemble
4. **Top 10 Factors**: Highest weighted factors
5. **Weight Distribution**: Statistics on weight allocation

Example:

```
================================================================================
ENSEMBLE ALPHA MINING - PERFORMANCE REPORT
================================================================================

TRAINING WINDOW SUMMARIES
--------------------------------------------------------------------------------

Window: 12m
  Period: 2023-01-01 to 2023-12-31
  Factors: 20
  stage2: 20 factors, IC=0.0456

Window: 6m
  Period: 2023-07-01 to 2023-12-31
  Factors: 20
  stage2: 20 factors, IC=0.0512

Window: 3m
  Period: 2023-10-01 to 2023-12-31
  Factors: 20
  stage2: 20 factors, IC=0.0489

ENSEMBLE SUMMARY
--------------------------------------------------------------------------------

Total factors collected: 60
Final ensemble size: 42
Validation IC: 0.0538

WINDOW CONTRIBUTIONS TO FINAL ENSEMBLE
--------------------------------------------------------------------------------

12m:
  Total factors: 20
  In final ensemble: 15
  Percentage: 75.0%

6m:
  Total factors: 20
  In final ensemble: 18
  Percentage: 90.0%

3m:
  Total factors: 20
  In final ensemble: 9
  Percentage: 45.0%
```

## Advanced Usage

### Custom Training Callback

Modify `EnsembleTrainingCallback` in [scripts/train_ensemble.py](../scripts/train_ensemble.py:229) to add:

- Custom logging
- Additional metrics
- Tensorboard integration
- Early stopping criteria

### Custom Operators

Add custom operators to stage-specific training:

```python
def get_operators_for_stage(stage: str, fundamental_features: List[str]) -> List:
    if stage == 'technical':
        return Operators  # Technical only

    elif stage == 'fundamentals':
        extended_ops = list(Operators)
        # Add your custom operators
        extended_ops.append(MyCustomOperator)
        return extended_ops
```

### Cross-Validation Enhancement

The current CV applies shrinkage. For time-based k-fold CV:

```python
def optimize_with_cross_validation(pool, config, logger):
    # Split validation period into k folds
    # Train on k-1, validate on 1
    # Average weights across folds
    # Apply shrinkage
    ...
```

See [scripts/train_ensemble.py:553](../scripts/train_ensemble.py:553)

### Parallel Window Training

For faster execution, train windows in parallel using:

```bash
# Window 1
python scripts/train_single_window.py --window 12m &

# Window 2
python scripts/train_single_window.py --window 6m &

# Window 3
python scripts/train_single_window.py --window 3m &

wait

# Merge
python scripts/merge_ensemble.py
```

## Troubleshooting

### Out of Memory

**Symptom**: CUDA out of memory error

**Solution**:
1. Reduce `pool_capacity_per_window` in config
2. Use `device: "cpu"` instead of GPU
3. Reduce `max_episodes`

### Low IC on Validation

**Symptom**: Validation IC << training IC

**Solution**:
1. Increase `shrinkage_factor` (e.g., 0.8)
2. Increase `l1_alpha` regularization
3. Use more conservative `ic_lower_bound`
4. Add more training windows for diversity

### Cache Not Working

**Symptom**: Always retrains despite cache

**Solution**:
1. Check `cache.enabled: true` in config
2. Verify cache files exist in `output/cache/`
3. Check universe hasn't changed
4. Ensure files are < `freshness_days` old

### Import Errors

**Symptom**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure alphagen is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/alphagen

# Or install in development mode
pip install -e .
```

## Best Practices

### 1. Start Small
- Test with 100 episodes per window
- Verify IC improvements
- Scale up gradually

### 2. Monitor Progress
- Watch tensorboard logs: `tensorboard --logdir output/ensemble/`
- Check early stopping triggers
- Review intermediate pool files

### 3. Validate Assumptions
- Run test suite after config changes
- Verify data availability for all windows
- Check fundamental data completeness

### 4. Iterative Refinement
- Analyze which windows contribute most
- Adjust window lengths based on regime changes
- Experiment with feature subsets

### 5. Production Deployment
- Use cache for consistent results
- Version control config files
- Archive final ensembles with dates

## Integration with Lean

To use the ensemble in QuantConnect Lean:

1. **Export factors**:

```python
from alphagen_lean.converter import convert_to_lean

with open('output/ensemble/ensemble_pool_2024Q1.json') as f:
    ensemble = json.load(f)

lean_code = convert_to_lean(ensemble)
with open('MyAlphaModel.py', 'w') as f:
    f.write(lean_code)
```

2. **Use in algorithm**:

```python
class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.AddAlpha(EnsembleAlphaModel())
```

## Performance Benchmarks

Typical execution times (NVIDIA A100):

| Stage | Duration | Notes |
|-------|----------|-------|
| 12M training | ~2 hours | 500 episodes |
| 6M training | ~1.5 hours | 500 episodes |
| 3M training | ~1 hour | 500 episodes |
| Ensemble merge | ~5 minutes | 60 factors |
| **Total** | **~5 hours** | Full run |

With caching enabled, subsequent runs: ~10 minutes

## References

- **AlphaGen Paper**: [Link to paper]
- **Qlib Documentation**: https://qlib.readthedocs.io/
- **Ensemble Methods**: Breiman (1996), "Stacked Generalization"
- **Regularization**: Tibshirani (1996), "The LASSO"

## Support

For issues or questions:

1. Check [TROUBLESHOOTING](#troubleshooting) section
2. Run test suite: `python scripts/test_ensemble.py`
3. Review logs in `output/ensemble/ensemble_training.log`
4. Open an issue on GitHub with:
   - Config file
   - Error message
   - Log excerpt

---

**Version**: 1.0
**Last Updated**: 2024-11-02
**Maintainer**: AlphaGen Team

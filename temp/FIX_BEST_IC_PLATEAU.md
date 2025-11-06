# Fix: Best IC Training Plateau Issue

## Problem Description

During Stage 1 training (e.g., "12m stage 1"), the `best_ic_ret` value was not improving as episodes progressed and remained stuck at 0 or other low values.

## Root Causes Identified

1. **Silent NaN failures**: When IC calculations returned NaN values, the code converted them to 0.0 instead of returning None, causing the objective to be artificially low and preventing `best_ic_ret` updates.

2. **Lack of visibility**: No logging or debugging information was available to understand:
   - Why expressions were failing IC calculations
   - What objective values were being computed
   - How often `best_ic_ret` was actually being updated

3. **NaN propagation**: In `CustomRewardAlphaPool._calc_main_objective()`, when `batch_pearsonr` returned NaN values, they were silently converted to 0.0, masking the underlying data quality issues.

## Changes Made

### 1. Enhanced Failure Tracking (`linear_alpha_pool.py:61-81`)

Added detailed tracking of WHY expressions fail IC calculations:
- `ic_ret is None`
- `ic_mut is None (high correlation or below threshold)`
- `ic_ret is NaN`
- `ic_mut contains NaN`

Failures are now stored in `_failure_stats` dictionary for analysis.

### 2. Improved Objective Tracking (`linear_alpha_pool.py:171-190`)

The `_maybe_update_best()` method now:
- Tracks all IC and objective values in `_obj_history`
- Counts successful updates in `_best_update_count`
- Provides visibility into the update process

### 3. Better NaN Handling (`linear_alpha_pool.py:457-521`)

`CustomRewardAlphaPool._calc_main_objective()` now:
- Returns `None` instead of 0.0 when critical NaN values are detected
- Catches exceptions and returns `None` to fall back to simple IC calculation
- Stores objective components (IC, ICIR, turnover penalty) for debugging
- Tracks errors in `_objective_errors` list

Key changes:
```python
# Before: Silent NaN conversion to 0
if math.isnan(ensemble_ic):
    ensemble_ic = 0.0

# After: Return None to signal failure
if not math.isfinite(ensemble_ic):
    return None
```

### 4. Debug Statistics API (`linear_alpha_pool.py:309-402`)

Added two new methods:

**`get_debug_stats() -> Dict[str, Any]`**
Returns comprehensive debugging information:
- Pool state (size, capacity, eval count)
- Best IC and objective values
- Failure statistics by reason
- Objective history and statistics
- Recent objective components (IC, ICIR, turnover)
- Error logs

**`print_debug_stats() -> None`**
Prints human-readable debugging information to console.

### 5. Integrated Debugging in Training (`train_ensemble.py:751-784`)

Modified `EarlyStoppingCallback._on_rollout_end()` to:
- Print debug stats every 10 episodes
- Show key metrics: best IC, pool size, update count, failures
- Display latest objective components
- Print full debug stats when early stopping triggers

## How to Use

### During Training

The training script now automatically prints debug information every 10 episodes:
```
[Episode 10] Debug Stats:
  Best IC: 0.023456, Best Obj: 0.045678
  Pool Size: 8/10
  Best Updates: 3, Total Evals: 1234
  Total Failures: 456 (37.0%)
    - ic_mut is None (high correlation or below threshold): 234
    - ic_ret is NaN: 122
  Latest Objective: IC=0.0234, ICIR=0.0456, Turnover=0.0123, Final=0.0567
```

### Manual Debugging

You can manually check pool statistics:

```python
# In your code or interactive session
pool = env.unwrapped.pool

# Get statistics as a dictionary
stats = pool.get_debug_stats()
print(stats)

# Or print human-readable format
pool.print_debug_stats()
```

### Understanding the Output

**Failure Statistics**: Shows why expressions are being rejected
- High `ic_mut is None` count → expressions are too correlated with existing factors
- High `ic_ret is NaN` count → data quality issues or numerical instability

**Objective Statistics**: Shows the range of objective values
- If `max` is only slightly above `best_obj`, the optimization is working but progress is slow
- If `max` equals `best_obj`, no new expressions are better than existing ones

**Objective Components**: Shows the breakdown of the custom reward
- `IC`: Ensemble information coefficient
- `ICIR`: IC Information Ratio (IC mean / IC std)
- `Turnover`: Penalty for high day-over-day changes
- `Final`: The combined objective = IC + ICIR - Turnover

## Expected Behavior After Fix

1. **Better visibility**: You'll now see exactly why expressions are failing and what objective values are being computed.

2. **Proper NaN handling**: When NaN values occur, the system will fall back to simple IC calculation instead of using corrupted 0.0 values.

3. **Actionable insights**: You can identify:
   - If the pool is saturated (high correlation rejections)
   - If there are data quality issues (high NaN rates)
   - If the objective function is working correctly
   - Whether training is actually stuck or just slow

4. **Early intervention**: You can adjust hyperparameters or data preprocessing based on the debug output.

## Next Steps

1. **Run training** with these changes and monitor the debug output
2. **Analyze failure patterns** to understand bottlenecks
3. **Adjust ic_mut_threshold** if too many expressions are rejected for high correlation
4. **Check data quality** if NaN rates are high
5. **Tune objective weights** (turnover_penalty_coeff) if needed based on component breakdown

## Files Modified

- `alphagen/models/linear_alpha_pool.py`: Core fixes for tracking and NaN handling
- `scripts/train_ensemble.py`: Integrated debugging output in training callback

# Best IC Training - Quick Reference Guide

## Where best_ic_ret is Tracked

### Key Variables
| Variable | Location | Initial Value | Purpose |
|----------|----------|----------------|---------|
| `best_ic_ret` | `alphagen/models/alpha_pool.py:21` | `-1.0` | Stores best IC found |
| `best_obj` | `alphagen/models/linear_alpha_pool.py:31` | `-1.0` | Stores best objective score |

### Update Flow
```
Pool Creation
  ↓
Expression Evaluation → try_new_expr()
  ├─ _calc_ics() → Check if IC is valid
  │  ├─ Returns None/NaN → return 0 (NO UPDATE)
  │  └─ Valid → continue
  ├─ _add_factor() → Add to pool
  ├─ optimize() → Optimize weights
  ├─ calculate_ic_and_objective() → Get IC and objective
  │  ├─ ic = evaluate_ensemble()
  │  ├─ obj = _calc_main_objective() (can be None)
  │  └─ if obj is None: obj = ic
  └─ _maybe_update_best(ic, obj) → UPDATE LOGIC HERE
     ├─ if obj > best_obj → UPDATE (best_obj = obj, best_ic_ret = ic)
     └─ else → NO UPDATE
```

## Critical Update Condition
```python
def _maybe_update_best(self, ic: float, obj: float) -> bool:
    if obj <= self.best_obj:  # <-- CRITICAL LINE
        return False
    self.best_obj = obj
    self.best_ic_ret = ic
    return True
```

**Key Point**: If `obj <= -1.0`, the update is SKIPPED!

## 5 Ways best_ic_ret Can Fail to Update

### 1. IC Calculation Fails (MOST COMMON)
**Location**: `linear_alpha_pool.py:64`
```python
if ic_ret is None or ic_mut is None or np.isnan(ic_ret):
    return 0.  # Returns without updating best_ic_ret!
```
**Cause**: 
- OutOfDataRangeError during expression evaluation
- High mutual correlation (>0.99)
- NaN in IC calculation

### 2. Objective is NaN
**Location**: `linear_alpha_pool.py:157`
```python
if obj <= self.best_obj:  # NaN <= -1.0 is False
    return False  # But NaN comparisons work differently!
```
**Cause**: 
- CustomRewardAlphaPool calculates: `IC + ICIR - turnover_penalty`
- If batch_pearsonr fails, result is NaN
- NaN propagates to best_ic_ret

### 3. Weight Optimization Fails Silently
**Location**: `linear_alpha_pool.py:335-337`
```python
def _optimize_lstsq(self):
    try:
        return np.linalg.lstsq(...)
    except:
        return self.weights  # Falls back silently!
```
**Cause**: Singular matrix or numerical issues

### 4. Expressions Immediately Removed as Worst
**Location**: `linear_alpha_pool.py:74-80`
```python
if worst_idx == self.capacity:
    self._pop(worst_idx)
    return self.best_obj  # Returns without updating!
```
**Cause**: New expression is worst in pool

### 5. Expression in Failure Cache
**Location**: `linear_alpha_pool.py:66`
```python
if str(expr) in self._failure_cache:
    return self.best_obj  # Returns without updating!
```
**Cause**: Same expression tried multiple times

## Stage 1 Training Flow
**File**: `scripts/train_ensemble.py:779-922`

```
1. Create pool (Line 845):
   CustomRewardAlphaPool with turnover_penalty_coeff=0.1

2. Environment training (Line 901):
   model.learn() with PPO

3. Callback monitoring (Line 740-762):
   - Gets pool.best_ic_ret
   - Checks if improved by epsilon (1e-5)
   - Early stopping if no improvement for 30 episodes

4. Result (Line 920):
   Logs "Stage 1 complete: IC={pool.best_ic_ret:.4f}"
```

## Debugging Symptoms

| Symptom | Most Likely Cause | How to Check |
|---------|------------------|--------------|
| best_ic_ret = -1.0 | No expressions passed validation | Check if pool.size > 0 |
| best_ic_ret = 0 | Objective calculation returns 0 or fails | Print objective values |
| best_ic_ret = NaN | NaN in objective calculation | Check batch_pearsonr output |
| best_ic_ret plateaus | Weight optimization fails | Add logging to optimize() |
| Training stops improving | Expression evaluation errors | Check for OutOfDataRangeError |

## Critical Files to Fix

### Priority 1: Add NaN protection
**File**: `alphagen/models/linear_alpha_pool.py:156-161`
```python
def _maybe_update_best(self, ic: float, obj: float) -> bool:
    if not np.isfinite(obj):  # Add this!
        return False
    if obj <= self.best_obj:
        return False
    self.best_obj = obj
    self.best_ic_ret = ic
    return True
```

### Priority 2: Better objective initialization
**File**: `alphagen/models/linear_alpha_pool.py:31`
```python
# Change from:
self.best_obj = -1.

# To:
self.best_obj = -float('inf')
```

### Priority 3: Add logging to optimization
**File**: `alphagen/models/linear_alpha_pool.py:335-337`
```python
def _optimize_lstsq(self) -> np.ndarray:
    try:
        return np.linalg.lstsq(...)
    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"lstsq failed: {e}")
        return self.weights
```

## Quick Verification

To verify best_ic_ret is updating correctly:

```python
pool = CustomRewardAlphaPool(...)

# Check initial state
print(f"Initial: best_ic_ret={pool.best_ic_ret}, best_obj={pool.best_obj}")

# After some training
print(f"After training: best_ic_ret={pool.best_ic_ret}, size={pool.size}")
print(f"Objective: best_obj={pool.best_obj}")

# Check for NaN
if np.isnan(pool.best_ic_ret):
    print("ERROR: best_ic_ret is NaN!")
```

## Configuration Parameters

In ensemble config:
```yaml
training:
  stage1_technical:
    pool_capacity_per_window: 100
    
  optimizer:
    l1_alpha: 0.005
    
  turnover_penalty_coeff: 0.1  # <-- Can cause negative objectives!
```

High `turnover_penalty_coeff` can make objective negative!

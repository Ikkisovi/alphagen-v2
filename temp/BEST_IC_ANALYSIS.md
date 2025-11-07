# Best IC Training Tracking Analysis

## Overview
Analysis of how best_ic_ret is tracked and updated during training, focusing on the Stage 1 training process and identifying potential issues causing best_ic to remain at 0 or -1.

---

## Key Files and Locations

### 1. **Alpha Pool Base Definition**
**File**: `/home/user/alphagen-v2/alphagen/models/alpha_pool.py` (Lines 1-44)

**Key Code**:
```python
class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(self, capacity: int, calculator: AlphaCalculator, device: torch.device = ...):
        self.size = 0
        self.capacity = capacity
        self.calculator = calculator
        self.device = device
        self.eval_cnt = 0
        self.best_ic_ret: float = -1.  # <-- INITIALIZED TO -1.0 HERE
```

**Issue**: `best_ic_ret` is initialized to `-1.0`, which is critical for the update logic.

---

### 2. **Linear Alpha Pool - Best IC Update Logic**
**File**: `/home/user/alphagen-v2/alphagen/models/linear_alpha_pool.py`

#### A. Initialization (Line 31)
```python
class LinearAlphaPool(AlphaPoolBase, metaclass=ABCMeta):
    def __init__(self, capacity: int, calculator: AlphaCalculator, ic_lower_bound: Optional[float] = None, ...):
        super().__init__(capacity, calculator, device)
        # ...
        self.best_obj = -1.  # <-- CRITICAL: Objective tracking starts at -1.0
```

#### B. Core Update Mechanism (Lines 156-161)
```python
def _maybe_update_best(self, ic: float, obj: float) -> bool:
    if obj <= self.best_obj:  # <-- CRITICAL COMPARISON
        return False
    self.best_obj = obj
    self.best_ic_ret = ic  # <-- best_ic_ret is updated here
    return True
```

**Problem**: If `obj <= best_obj (-1.0)`, the update is skipped. This can happen when:
- Initial objective values are negative
- Objective function returns values close to or below -1.0
- Due to floating-point precision issues

#### C. Objective Calculation (Lines 146-151)
```python
def calculate_ic_and_objective(self) -> Tuple[float, float]:
    ic = self.evaluate_ensemble()  # Get ensemble IC
    obj = self._calc_main_objective()  # Get custom objective
    if obj is None:
        obj = ic  # Fall back to IC if no custom objective
    return ic, obj
```

**Critical Path**: For `CustomRewardAlphaPool` (used in ensemble training), `_calc_main_objective()` returns:
```python
final_objective = ensemble_ic + ensemble_icir - turnover_penalty
```

This objective can be **negative** if turnover_penalty is high!

#### D. Ensemble Evaluation (Lines 170-173)
```python
def evaluate_ensemble(self) -> float:
    if self.size == 0:
        return 0.
    return self.calculator.calc_pool_IC_ret(self.exprs[:self.size], self.weights)
```

---

### 3. **Stage 1 Training - try_new_expr Flow**
**File**: `/home/user/alphagen-v2/alphagen/models/linear_alpha_pool.py` (Lines 61-104)

**Three paths that SKIP `_maybe_update_best` call**:

1. **Path 1: Early IC Validation Failure (Line 64)**
   ```python
   if ic_ret is None or ic_mut is None or np.isnan(ic_ret) or np.isnan(ic_mut).any():
       return 0.  # <-- RETURNS WITHOUT UPDATING best_ic_ret
   ```
   - Returns if IC calculation fails
   - Common causes: OutOfDataRangeError, NaN values, high correlation threshold (>0.99)

2. **Path 2: Failure Cache (Line 66)**
   ```python
   if str(expr) in self._failure_cache:
       return self.best_obj  # <-- RETURNS WITHOUT UPDATING
   ```
   - Prevents retrying failed expressions

3. **Path 3: Worst Expression Removal (Lines 74-80)**
   ```python
   if self.size > self.capacity:  # Need to remove one
       worst_idx = int(np.argmin(np.abs(new_weights)))
       if worst_idx == self.capacity:
           self._pop(worst_idx)
           self._failure_cache.add(str(expr))
           return self.best_obj  # <-- RETURNS WITHOUT UPDATING
   ```
   - Returns if newly added expression is immediately removed as worst

**Successful Path** (Lines 102-103):
```python
new_ic_ret, new_obj = self.calculate_ic_and_objective()
self._maybe_update_best(new_ic_ret, new_obj)  # <-- UPDATE HAPPENS HERE
```

---

### 4. **Warm-Start Loading - force_load_exprs**
**File**: `/home/user/alphagen-v2/alphagen/models/linear_alpha_pool.py` (Lines 106-144)

```python
def force_load_exprs(self, exprs: List[Expression], weights: Optional[List[float]] = None) -> None:
    old_ic = self.evaluate_ensemble()
    # ... Loop through expressions, skip invalid ones ...
    self.weights = self.optimize()  # <-- Optimize weights
    new_ic, new_obj = self.calculate_ic_and_objective()
    self._maybe_update_best(new_ic, new_obj)  # <-- UPDATE HAPPENS HERE (Line 137)
```

**Critical**: The `optimize()` call can fail silently in `MseAlphaPool._optimize_lstsq()` if matrix is singular.

---

### 5. **Stage 1 Training Entry Point**
**File**: `/home/user/alphagen-v2/scripts/train_ensemble.py` (Lines 779-922)

**Key Configuration (Lines 823-853)**:
```python
stage1_config = training_config['stage1_technical']
pool = CustomRewardAlphaPool(
    capacity=training_config['pool_capacity_per_window'],
    calculator=calculator,
    ic_lower_bound=config['ensemble'].get('ic_lower_bound', 0.01),
    l1_alpha=config['ensemble']['optimizer']['l1_alpha'],
    turnover_penalty_coeff=training_config.get('turnover_penalty_coeff', 0.1),  # <-- High penalty can cause negative objectives
    device=device
)
```

**Monitoring (Lines 744-762)**:
```python
current_ic = pool.best_ic_ret
self.logger.record(f'{self.window_name}/best_ic', current_ic)

if current_ic > self.best_ic + 1e-5:  # <-- Early stopping check
    self.best_ic = current_ic
    self.episodes_since_improvement = 0
else:
    self.episodes_since_improvement += 1
```

---

## Root Cause Analysis

### Primary Issue: Negative Objective Function Values

**Problem Chain**:
1. `best_obj` initialized to `-1.0`
2. `CustomRewardAlphaPool` objective = `IC + ICIR - turnover_penalty`
3. With high `turnover_penalty_coeff`, objective can be **negative**
4. Example: `IC=0.01 + ICIR=0.05 - turnover_penalty=0.5 = -0.44`
5. Since `-0.44 <= -1.0` is False... wait, that's true, so it should update!

**Actually**: Let me reconsider:
- `-0.44 > -1.0` is True, so it SHOULD update
- BUT: If all objectives are between -1.0 and 0, and start at -1.0:
  - First expression: `obj = -0.5`, check `-0.5 > -1.0` → True → UPDATE ✓
  - So this isn't the issue...

### Secondary Issue: Very Low or Invalid IC Values

**True Problem**:
1. If many expressions fail IC calculation:
   - `_calc_ics()` returns None or NaN
   - These return early at line 64
   - No update to `best_ic_ret` occurs

2. If no valid expressions are added to pool:
   - `best_ic_ret` stays at initial value of `-1.0`
   - Pool size remains 0
   - All subsequent calls to `evaluate_ensemble()` return `0.0`
   - But `best_obj = -1.0` and objective `obj = 0.0 > -1.0` → Should update!

### Tertiary Issue: Objective Initialization Timing

**The Real Issue**:
When the pool is created and first expression is added:
1. `pool.best_ic_ret = -1.0`, `pool.best_obj = -1.0`
2. First expression added: `ic = 0.05, obj = 0.0`
3. Check: `0.0 <= -1.0`? No, `0.0 > -1.0` is True
4. Should update... but what if the objective calculation fails?

Looking at `CustomRewardAlphaPool._calc_main_objective()`:
```python
if self.size == 0:
    return 0.0  # <-- Good, avoids errors
# ... calculations that could return NaN or inf ...
final_objective = ensemble_ic + ensemble_icir - turnover_penalty
return final_objective
```

If `ensemble_icir` or `ensemble_ic` is NaN (from batch_pearsonr failing):
- `NaN + X - Y = NaN`
- Then in `calculate_ic_and_objective()`: `obj = NaN`
- Then `NaN <= -1.0` is False → NO UPDATE (NaN comparisons are always False!)

---

## Identified Issues

### Issue #1: NaN Propagation in Objective (CRITICAL)
**Location**: `CustomRewardAlphaPool._calc_main_objective()`, line 435-468

**Problem**: If batch_pearsonr returns NaN:
- `ensemble_ic` becomes NaN
- `ensemble_icir` becomes NaN
- `final_objective` becomes NaN
- In `_maybe_update_best`: `NaN <= -1.0` is False, but also `NaN > -1.0` is False!
- Actually, the condition is `if obj <= self.best_obj: return False`, and `NaN <= -1.0` is False
- So it doesn't return early, it CONTINUES and updates with NaN!

Actually, wait. Let me re-read line 157:
```python
if obj <= self.best_obj:
    return False
```

With NaN:
- `NaN <= -1.0` evaluates to False
- So `return False` is NOT executed
- We continue to line 159-160 and SET `best_obj = NaN, best_ic_ret = ic`

So if `ic` is also NaN, then `best_ic_ret = NaN`!

### Issue #2: Weight Optimization Failure (CRITICAL)
**Location**: `MseAlphaPool._optimize_lstsq()`, line 335-337

```python
def _optimize_lstsq(self) -> np.ndarray:
    try:
        return np.linalg.lstsq(self._mutual_ics[:self.size, :self.size], self.single_ics[:self.size])[0]
    except (np.linalg.LinAlgError, ValueError):
        return self.weights  # <-- Returns old weights silently
```

**Problem**: If optimization fails:
- Old weights are returned
- IC calculation uses failed weights
- Pool becomes stuck with poor IC
- No feedback to user about optimization failure

### Issue #3: Expression Evaluation Errors (CRITICAL)
**Location**: `AlphaEnvCore._evaluate()`, line 67-76

```python
def _evaluate(self):
    expr: Expression = self._builder.get_tree()
    try:
        ret = self.pool.try_new_expr(expr)
        self.eval_cnt += 1
        return ret
    except OutOfDataRangeError:
        return 0.  # <-- Fails silently, no pool update
```

**Problem**: `OutOfDataRangeError` returns 0 reward without updating pool

### Issue #4: Comparison Logic Edge Case
**Location**: `_maybe_update_best()`, line 157

```python
if obj <= self.best_obj:  # Should this be '<' instead of '<='?
    return False
```

**Problem**: With `best_obj = -1.0`:
- If `obj = -1.0` exactly, `obj <= best_obj` is True, so we don't update
- This creates a "stuck" state where tie-breaks don't trigger updates
- Should be `if obj <= best_obj:` only if tie-breaking is explicitly not wanted

---

## Summary Table

| Issue | Location | Severity | Impact | Symptoms |
|-------|----------|----------|--------|----------|
| NaN in objective | CustomRewardAlphaPool._calc_main_objective | CRITICAL | Undetected NaN propagation | best_ic_ret becomes NaN |
| Silent weight optimization failure | MseAlphaPool._optimize_lstsq | CRITICAL | Pool stuck in local minimum | IC plateaus at 0 |
| Expression evaluation errors not reported | AlphaEnvCore._evaluate | CRITICAL | No feedback on data issues | Training stops improving |
| Comparison uses <= instead of < | LinearAlphaPool._maybe_update_best | MEDIUM | Tie-breaks don't update | Potential stuck state |
| bad_obj init value | LinearAlphaPool.__init__ | MEDIUM | Depends on objective function | Can miss valid objectives |

---

## Recommendations for Fixes

1. **Add NaN checking before objective update**:
   ```python
   def _maybe_update_best(self, ic: float, obj: float) -> bool:
       if not np.isfinite(obj):  # Check for NaN, inf, -inf
           return False
       if obj <= self.best_obj:
           return False
       self.best_obj = obj
       self.best_ic_ret = ic
       return True
   ```

2. **Log weight optimization failures**:
   ```python
   def _optimize_lstsq(self) -> np.ndarray:
       try:
           return np.linalg.lstsq(...)[0]
       except (np.linalg.LinAlgError, ValueError) as e:
           logger.warning(f"Weight optimization failed: {e}, using previous weights")
           return self.weights
   ```

3. **Improve expression evaluation error reporting**:
   ```python
   def _evaluate(self):
       try:
           ret = self.pool.try_new_expr(expr)
           return ret
       except OutOfDataRangeError as e:
           logger.warning(f"Expression evaluation out of data range: {e}")
           return 0.
   ```

4. **Initialize best_obj to negative infinity**:
   ```python
   self.best_obj = -float('inf')  # Instead of -1.0
   ```


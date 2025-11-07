# AlphaGen v2 vs Original AlphaGen - Comprehensive Comparison

**Date:** 2025-11-07
**Analysis:** Detailed comparison to identify performance degradation causes

---

## 🔴 CRITICAL DIFFERENCES CAUSING POOR PERFORMANCE

### 1. REWARD FUNCTION - MAJOR DIFFERENCE ⚠️

#### Original AlphaGen:
```python
# File: alphagen/models/linear_alpha_pool.py:145-146
def _calc_main_objective(self) -> Optional[float]:
    "Get the main optimization objective, return None for the default (ensemble IC)."
    # Returns None by default → uses ensemble IC as reward
    # Reward = Ensemble IC (range: -1 to 1)
```

#### Your AlphaGen v2:
```python
# File: /home/user/alphagen-v2/alphagen/models/linear_alpha_pool.py:617-673
def _calc_main_objective(self) -> Optional[float]:
    # Complex reward: IC + ICIR - turnover_penalty
    ensemble_ic = ics.mean().item()
    ensemble_icir = (ic_mean / ic_std).item()  # Can be very large!
    turnover = self._compute_selection_turnover(weighted_alpha)
    turnover_penalty = self.turnover_penalty_coeff * turnover
    final_objective = ensemble_ic + ensemble_icir - turnover_penalty
    return final_objective
```

**Config:**
```yaml
# config/nov2025_ensemble_config.yaml:71-77
reward:
  objective: "IC + ICIR - turnover_penalty"
  turnover_penalty_coeff: 0.05
  turnover_rebalance_horizon: 22
  turnover_top_k_ratio: 0.1
```

**Impact:**
- ❌ Original: Simple ensemble IC (range: -1 to 1, stable signal)
- ❌ Yours: IC + ICIR - penalty (unbounded, unstable signal)
- ❌ ICIR = IC_mean / IC_std can be very large when std is small
- ❌ Turnover penalty adds noise to optimization
- ❌ **RL agent receives unstable reward signal → poor training**

---

### 2. DATA HANDLING - CRITICAL DIFFERENCE ⚠️

#### Original AlphaGen:
```python
# File: scripts/rl.py:206-213
segments = [
    ("2012-01-01", "2021-12-31"),  # 10 YEARS of training data
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30")
]
# Uses DAILY frequency data from Qlib
```

#### Your AlphaGen v2:
```yaml
# config/nov2025_ensemble_config.yaml:7-26
time_windows:
  train_12m:
    start_date: "2024-11-01"
    end_date: "2025-10-31"      # Only 12 months!
  train_6m:
    start_date: "2025-05-01"
    end_date: "2025-10-31"      # Only 6 months!
  train_3m:
    start_date: "2025-08-01"
    end_date: "2025-10-31"      # Only 3 months!
```

**Impact:**
- ❌ Original: **10 years** of training data (2012-2021)
- ❌ Yours: **12 months maximum** (dramatically less)
- ❌ Less data → **Less robust IC estimates** → Worse factor quality
- ❌ Your AM/PM data: Each day split into 2 periods → **Even fewer samples per period**
- ❌ **IC calculation unreliable with insufficient data**

---

### 3. TRAINING HYPERPARAMETERS - CRITICAL DIFFERENCE ⚠️

#### Original AlphaGen:
```python
# File: scripts/rl.py:216-225, 300-305
pool = MseAlphaPool(
    capacity=pool_capacity,      # Typically 10-20
    calculator=calculators[0],
    ic_lower_bound=None,         # No lower bound filtering
    l1_alpha=5e-3,
    device=device
)

default_steps = {
    10: 200_000,
    20: 250_000,    # 250k steps for pool capacity 20
    50: 300_000,
    100: 350_000
}
```

#### Your AlphaGen v2:
```yaml
# config/nov2025_ensemble_config.yaml:54-80
training:
  stage1_technical:
    max_episodes: 100      # Very low! (episode = rollout)
    min_episodes: 50
    early_stopping_patience: 15
    target_candidates: 40   # High compared to original

  pool_capacity_per_window: 20

  ppo:
    learning_rate: 0.0001
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    entropy_coef: 0.01
    value_loss_coef: 0.5
```

**Impact:**
- ❌ Original: **250,000 steps** for pool capacity 20
- ❌ Yours: **~100 episodes** (unclear total steps, but much less)
- ❌ **Insufficient exploration** → RL doesn't find good factors
- ❌ Early stopping may terminate training prematurely

---

### 4. POOL ARCHITECTURE DIFFERENCE

#### Original AlphaGen:
```python
# Uses MseAlphaPool (Mean Squared Error optimization)
# Objective: Maximize ensemble IC
# Simple, stable optimization
pool = MseAlphaPool(
    capacity=pool_capacity,
    calculator=calculators[0],
    ic_lower_bound=None,
    l1_alpha=5e-3,
    device=device
)
```

#### Your AlphaGen v2:
```python
# Uses CustomRewardAlphaPool (extends MeanStdAlphaPool)
# Objective: IC + ICIR - turnover_penalty
# Complex, multi-objective optimization
pool = CustomRewardAlphaPool(
    capacity=capacity,
    calculator=calculator,
    ic_lower_bound=ic_lower_bound,
    l1_alpha=l1_alpha,
    lcb_beta=None,
    turnover_penalty_coeff=turnover_penalty_coeff,
    turnover_rebalance_horizon=turnover_rebalance_horizon,
    turnover_top_k_ratio=turnover_top_k_ratio,
    device=device
)
```

**Impact:**
- ❌ Original: Single objective (IC) → Clear optimization target
- ❌ Yours: Multiple objectives → Competing signals, harder to optimize

---

### 5. TENSORBOARD LOGGING ISSUE 🔍

**Why you think "tensorboard doesn't work":**

Your code DOES log to tensorboard:
```python
# scripts/train_ensemble.py:850
tensorboard_log=str(output_dir / 'tensorboard_stage1')

# scripts/train_ensemble.py:659-676
self.logger.record(f'{self.window_name}/best_ic_step', pool.best_ic_ret)
self.logger.dump(step=self.num_timesteps)
```

**The issue:**
1. ✅ Tensorboard logs ARE being written
2. ❌ Location: `output/nov2025_ensemble/tensorboard_stage1/`
3. ❌ You need to check the correct directory
4. ❌ Best IC is logged every 2000 steps, not every step

**To view tensorboard:**
```bash
tensorboard --logdir output/nov2025_ensemble/tensorboard_stage1
```

**Metrics logged:**
- `{window_name}/pool_size_step` (every 2000 steps)
- `{window_name}/best_ic_step` (every 2000 steps)
- `{window_name}/episode` (every rollout)
- `{window_name}/pool_size` (every rollout)
- `{window_name}/best_ic` (every rollout)

---

## 🎯 ROOT CAUSES SUMMARY

### Primary Issues (Fix These First):

1. **Complex Reward Signal**
   - Original: Simple IC (-1 to 1)
   - Yours: IC + ICIR - penalty (unbounded, unstable)
   - **Effect:** RL agent can't learn effectively

2. **Insufficient Training Data**
   - Original: 10 years
   - Yours: 12 months maximum
   - **Effect:** IC estimates unreliable, poor factor quality

3. **Insufficient Training Steps**
   - Original: 250,000 steps
   - Yours: ~100 episodes (much less)
   - **Effect:** Insufficient exploration, poor factors

4. **AM/PM Data Frequency**
   - Splits each day into 2 periods
   - **Effect:** Reduces sample size, adds noise

---

## 💡 RECOMMENDED FIXES

### CRITICAL (Fix Immediately):

#### 1. Simplify Reward to IC Only
Create new config: `config/simple_ic_config.yaml`
```yaml
reward:
  objective: "IC"  # Remove ICIR and turnover penalty
  # Comment out complex reward components
```

Or modify pool initialization:
```python
# Use MseAlphaPool instead of CustomRewardAlphaPool
pool = MseAlphaPool(
    capacity=pool_capacity,
    calculator=calculator,
    ic_lower_bound=0.01,  # Optional: filter weak factors
    l1_alpha=5e-3,
    device=device
)
```

#### 2. Increase Training Data to 3+ Years
```yaml
time_windows:
  train_12m:
    start_date: "2020-01-01"  # Use 5+ years instead of 1 year
    end_date: "2025-10-31"
  train_6m:
    start_date: "2020-01-01"  # Or use same long window
    end_date: "2025-10-31"
```

#### 3. Increase Training Steps
```yaml
training:
  stage1_technical:
    max_episodes: 1000  # Increase from 100 to 1000
    min_episodes: 500
    target_candidates: 20  # Reduce from 40 to 20
```

---

### IMPORTANT (Fix Soon):

#### 4. Switch from AM/PM to Daily Data
- Use daily frequency instead of AM/PM
- More samples per period → More reliable IC
- Original alphagen uses daily data

#### 5. Verify TensorBoard
```bash
# Check correct directory
ls output/nov2025_ensemble/tensorboard_stage1/

# Run tensorboard
tensorboard --logdir output/nov2025_ensemble/tensorboard_stage1
```

---

### NICE TO HAVE (Optimize Later):

#### 6. Remove Turnover Penalty Initially
```yaml
reward:
  objective: "IC"  # Start simple
  # Add turnover penalty back later after confirming IC works
```

#### 7. Start with Smaller Pool
```yaml
training:
  pool_capacity_per_window: 10  # Start with 10 instead of 20
```

---

## 🧪 QUICK TEST CONFIG

Create `config/test_simple_ic.yaml`:
```yaml
# Minimal config to test if simple IC works
time_windows:
  train_12m:
    name: "12m"
    start_date: "2020-01-01"  # 5 years of data
    end_date: "2025-10-31"

training:
  stage1_technical:
    max_episodes: 500  # More episodes
    target_candidates: 20  # Fewer, better factors

  reward:
    objective: "IC"  # Simple IC only

  pool_capacity_per_window: 10  # Small pool

  ppo:
    learning_rate: 0.0001
    gamma: 1.0  # Match original (gamma=1.0)
    ent_coef: 0.01
    batch_size: 128
```

Run test:
```bash
python scripts/train_ensemble.py --config-file config/test_simple_ic.yaml
```

This should give **significantly better IC results** (closer to original alphagen).

---

## 📊 EXPECTED IMPROVEMENTS

After implementing fixes:

**Before (Current):**
- Best IC: Low/unstable
- Poor performance on both AM/PM and daily data
- Manual printing required to see IC

**After (Expected):**
- Best IC: 0.05+ (depends on market conditions)
- Stable improvement over episodes
- TensorBoard shows clear training progress
- Factors perform well on test periods

---

## 🔍 DEBUGGING CHECKLIST

If performance still poor after fixes:

1. **Check IC calculation is correct**
   ```python
   # Verify IC calculation matches original
   ic = batch_pearsonr(alpha_values, target_values).mean()
   ```

2. **Verify data is loaded correctly**
   ```python
   # Print data shapes and date ranges
   print(f"Data shape: {stock_data.data.shape}")
   print(f"Date range: {stock_data.start_time} to {stock_data.end_time}")
   ```

3. **Monitor tensorboard during training**
   ```bash
   tensorboard --logdir output/nov2025_ensemble/tensorboard_stage1
   # Should see best_ic increasing over episodes
   ```

4. **Check pool statistics**
   ```python
   # Print pool state every 10 episodes
   print(f"Pool size: {pool.size}")
   print(f"Best IC: {pool.best_ic_ret}")
   print(f"Weights: {pool.weights}")
   ```

---

## 📝 IMPLEMENTATION PRIORITY

**Week 1:**
1. Switch to simple IC reward (remove ICIR and turnover penalty)
2. Increase training data to 3+ years
3. Increase training episodes to 500+
4. Run test and compare results

**Week 2:**
5. Switch from AM/PM to daily data (if needed)
6. Verify tensorboard monitoring works
7. Fine-tune hyperparameters

**Week 3:**
8. Consider adding turnover penalty back (if simple IC works)
9. Experiment with ensemble configurations
10. Optimize for production deployment

---

## 🚀 NEXT STEPS

Would you like me to:
1. ✅ Create a simplified config file with recommended settings?
2. ✅ Modify the pool class to use simple IC?
3. ✅ Set up proper tensorboard monitoring?
4. ✅ Create a test script to verify improvements?

**Recommendation:** Start with fix #1 (simple IC) and #2 (more data) to see immediate improvement.

---

## 📚 REFERENCES

**Original AlphaGen:**
- Repository: https://github.com/RL-MLDM/alphagen
- Main training script: `/tmp/original_alphagen/scripts/rl.py`
- Pool implementation: `/tmp/original_alphagen/alphagen/models/linear_alpha_pool.py`

**Your AlphaGen v2:**
- Main training script: `/home/user/alphagen-v2/scripts/train_ensemble.py`
- Pool implementation: `/home/user/alphagen-v2/alphagen/models/linear_alpha_pool.py`
- Current config: `/home/user/alphagen-v2/config/nov2025_ensemble_config.yaml`

---

**Analysis completed:** 2025-11-07
**Status:** Ready for implementation

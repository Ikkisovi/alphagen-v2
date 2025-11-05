"""
AlphaGen training script for Lean strategy.

This script is invoked from Lean via subprocess to train alpha factors and output results.
"""
#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from typing import Dict

# Add paths - this script is now in lean_project directory
# alphagen_lean module is also copied here
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
# Also add parent directory to access alphagen main package if available
alphagen_root = script_dir.parent if (script_dir.parent / "alphagen").exists() else script_dir
sys.path.insert(0, str(alphagen_root))

from alphagen.data.expression import Feature, Ref
from alphagen.local_data import LocalDataConfig, load_local_stock_data
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData
from sb3_contrib.ppo_mask import MaskablePPO

from alphagen_lean.data_prep import prepare_window_data


FUNDAMENTAL_FEATURES = {
    FeatureType.PE_RATIO,
    FeatureType.PB_RATIO,
    FeatureType.PS_RATIO,
    FeatureType.EV_TO_EBITDA,
    FeatureType.EV_TO_REVENUE,
    FeatureType.EV_TO_FCF,
    FeatureType.EARNINGS_YIELD,
    FeatureType.FCF_YIELD,
    FeatureType.SALES_YIELD,
    FeatureType.FORWARD_PE_RATIO,
    FeatureType.SHARES_OUTSTANDING,
    FeatureType.MARKET_CAP,
}
FUNDAMENTAL_TOKENS = {f"${ft.name.lower()}" for ft in FUNDAMENTAL_FEATURES}
DEFAULT_TEST_MONTHS = 1


def parse_args():
    """                     """
    parser = argparse.ArgumentParser(description="Train AlphaGen for Lean strategy")

    parser.add_argument("--ticker-pool", type=str, required=True,
                       help="JSON array of ticker symbols")
    parser.add_argument("--start-date", type=str, required=True,
                       help="Data start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="Data end date (YYYY-MM-DD)")
    parser.add_argument("--train-months", type=int, default=12,
                       help="Training window in months")
    parser.add_argument("--steps", type=int, default=5000,
                       help="PPO training steps")
    parser.add_argument("--pool-capacity", type=int, default=10,
                       help="Alpha pool capacity")
    parser.add_argument("--forward-horizon", type=int, default=20,
                       help="Forward return horizon in days")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Training device (cpu or cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no-seed", action="store_true",
                       help="Disable seeding fundamental features into the pool")
    parser.add_argument("--price-only-stage", action="store_true",
                       help="Train using only OHLCV features without fundamentals")
    parser.add_argument("--warm-start", type=str, default=None,
                       help="Path to prior factor JSON for warm start")

    return parser.parse_args()


def main():
    """Main training workflow."""
    args = parse_args()

    print("="*80)
    print("AlphaGen Training for Lean")
    print("="*80)
    print(f"Ticker pool: {args.ticker_pool[:100]}...")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Train months: {args.train_months}")
    print(f"Steps: {args.steps}")
    print(f"Pool capacity: {args.pool_capacity}")
    print(f"Forward horizon: {args.forward_horizon}")
    print(f"Output: {args.output}")
    print("="*80 + "\n")

    #       ticker pool
    ticker_pool = json.loads(args.ticker_pool)
    print(f"Parsed {len(ticker_pool)} tickers")

    device = torch.device(args.device)
    reseed_everything(args.seed)

    print("\n[1/5] Preparing data...")
    # Data path - use env variable or default
    # In Docker container, data is typically mounted at specific locations
    data_path_env = os.environ.get('LEAN_DATA_PATH', '/Data')
    data_path = Path(data_path_env) / "equity" / "usa"
    data_pickle = script_dir / "temp_train_data.pkl"

    print(f"  Data path: {data_path}")
    print(f"  Temp pickle: {data_pickle}")

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    try:
        df = prepare_window_data(
            data_path=data_path,
            symbols=ticker_pool,
            start_date=start_date,
            end_date=end_date,
            output_path=data_pickle
        )
        print(f"  Loaded {len(df)} records")
    except Exception as e:
        print(f"  ERROR: Failed to load data: {e}")
        return 1

    # Load StockData
    print("\n[2/5] Loading stock data...")
    fundamental_path = script_dir / "data" / "fundamentals" / "fundamentals.parquet"
    if not fundamental_path.exists():
        print(f"  Warning: Fundamental dataset not found at {fundamental_path}. Proceeding without fundamentals.")
        fundamental_path = None
    price_features = [
        FeatureType.OPEN,
        FeatureType.HIGH,
        FeatureType.LOW,
        FeatureType.CLOSE,
        FeatureType.VOLUME,
        FeatureType.VWAP,
    ]
    if args.price_only_stage:
        selected_features = price_features
        fundamental_path = None
    else:
        selected_features = list(FeatureType)

    cfg = LocalDataConfig(
        max_backtrack_days=120,
        max_future_days=args.forward_horizon + 10,
        device=device,
        features=selected_features,
        fundamental_path=fundamental_path,
    )

    fundamental_coverage: Dict[FeatureType, float] = {}

    try:
        stock = load_local_stock_data(data_pickle, config=cfg)
        print(f"  Dataset: {stock.n_days} days, {stock.n_stocks} stocks, {stock.n_features} features")
        if fundamental_path:
            valid_slice = slice(stock.max_backtrack_days, stock.max_backtrack_days + stock.n_days)
            valid_tensor = stock.data[valid_slice]
            print("  Fundamental feature coverage (non-NaN ratio):")
            for ft in sorted(FUNDAMENTAL_FEATURES, key=lambda f: f.name):
                idx = int(ft)
                values = valid_tensor[:, idx, :]
                coverage = torch.isfinite(values).float().mean().item()
                fundamental_coverage[ft] = coverage
                print(f"    {ft.name.lower():>18}: {coverage:6.2%}")
    except Exception as e:
        print(f"  ERROR: Failed to load StockData: {e}")
        return 1

    #             /         
    print("\n[3/5] Splitting train/test...")

    all_dates = pd.date_range(stock._start_time, stock._end_time, freq='D')
    if len(all_dates) == 0:
        print("  ERROR: No dates found in dataset.")
        return 1

    data_start = pd.to_datetime(all_dates[0])
    data_end = pd.to_datetime(all_dates[-1])

    test_end = data_end
    test_start = max(data_start, test_end - relativedelta(months=DEFAULT_TEST_MONTHS) + timedelta(days=1))
    train_end = test_start - timedelta(days=1)

    if train_end < data_start:
        print("  ERROR: Not enough history to create a train/test split. Provide more data or reduce --train-months.")
        return 1

    train_start = max(data_start, train_end - relativedelta(months=args.train_months) + timedelta(days=1))

    if train_start > train_end:
        print("  ERROR: Training window is empty after applying configuration.")
        return 1
    if test_start > test_end:
        print("  ERROR: Test window is empty after applying configuration.")
        return 1

    print(f"  Train: {train_start.date()} to {train_end.date()}")
    print(f"  Test: {test_start.date()} to {test_end.date()}")

    try:
        train_stock = stock[stock.find_date_slice(
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d")
        )]
        test_stock = stock[stock.find_date_slice(
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d")
        )]
        if train_stock.n_days <= 0:
            print("  ERROR: Training split returned no trading days.")
            return 1
        if test_stock.n_days <= 0:
            print("  ERROR: Test split returned no trading days.")
            return 1
        print(f"  Train: {train_stock.n_days} days")
        print(f"  Test: {test_stock.n_days} days")
    except Exception as e:
        print(f"  ERROR: Failed to slice data: {e}")
        return 1

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -args.forward_horizon) / close - 1

    train_calculator = QLibStockDataCalculator(train_stock, target)
    test_calculator = QLibStockDataCalculator(test_stock, target)

    #       Alpha   
    print("\n[4/5] Training AlphaGen...")
    print(f"  Creating Alpha Pool (capacity={args.pool_capacity}, l1_alpha=5e-3)")
    pool = MseAlphaPool(
        capacity=args.pool_capacity,
        calculator=train_calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device
    )

    if args.warm_start:
        try:
            warm_path = Path(args.warm_start)
            with open(warm_path, "r") as fh:
                warm_data = json.load(fh)
            expr_strings = warm_data.get("expressions", [])
            warm_exprs = []
            for expr_str in expr_strings:
                try:
                    warm_exprs.append(parse_expression(expr_str))
                except Exception as ex:
                    print(f"  Warning: Failed to parse warm-start expression {expr_str}: {ex}")
            if warm_exprs:
                warm_weights = warm_data.get("weights")
                if isinstance(warm_weights, list) and len(warm_weights) >= len(warm_exprs):
                    use_weights = warm_weights[:len(warm_exprs)]
                else:
                    use_weights = None
                print(f"  Warm-starting pool with {len(warm_exprs)} expressions from {warm_path}")
                pool.force_load_exprs(warm_exprs, use_weights)
        except Exception as ex:
            print(f"  Warning: Failed to warm-start from {args.warm_start}: {ex}")

    seeding_enabled = (not args.no_seed) and (not args.price_only_stage)
    if not seeding_enabled:
        print("  Seeding disabled via --no-seed or price-only stage")
    else:
        seed_features = [
            ft for ft in FUNDAMENTAL_FEATURES
            if fundamental_coverage.get(ft, 0.0) > 0.0
        ]
        if seed_features:
            max_seed = min(len(seed_features), max(args.pool_capacity // 2, 1))
            seed_exprs = [Feature(ft) for ft in seed_features[:max_seed]]
            seed_tokens = ", ".join(f"${ft.name.lower()}" for ft in seed_features[:max_seed])
            print(f"  Seeding pool with {max_seed} fundamental features: {seed_tokens}")
            pool.force_load_exprs(seed_exprs)

    #       RL      
    print(f"  Creating RL Environment")
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    #       PPO      
    print(f"  Creating PPO Model (LSTM, 2 layers, d_model=128)")
    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=2,
            d_model=128,
            dropout=0.1,
            device=device
        )
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        gamma=1.0,
        ent_coef=0.01,
        batch_size=128,
        n_steps=min(2048, args.steps),
        device=device,
        verbose=0
    )

    print(f"  Training for {args.steps} steps...")
    print(f"  Progress will be logged every {args.steps//10} steps")
    try:
        #       callback               
        from stable_baselines3.common.callbacks import BaseCallback

        class ProgressCallback(BaseCallback):
            def __init__(self, log_freq):
                super().__init__()
                self.log_freq = log_freq

            def _on_step(self):
                if self.n_calls % self.log_freq == 0:
                    print(f"    Step {self.n_calls}/{args.steps} | Pool IC: {pool.best_ic_ret:.4f} | Pool Size: {pool.size}")
                return True

        callback = ProgressCallback(log_freq=max(100, args.steps//10))
        model.learn(total_timesteps=args.steps, progress_bar=False, callback=callback)
    except Exception as e:
        print(f"  ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n[5/5] Evaluating and saving results...")
    train_ic = pool.best_ic_ret

    if pool.size > 0:
        test_ic, test_ric = pool.test_ensemble(test_calculator)
        expressions = [str(expr) for expr in pool.exprs[:pool.size]]
        weights = pool.weights.tolist()[:pool.size]
    else:
        print("  Warning: Alpha pool is empty after training; no factors to evaluate. Consider increasing steps or adjusting parameters.")
        test_ic = None
        test_ric = None
        expressions = []
        weights = []

    print(f"  Train IC: {train_ic:.4f}")
    if test_ic is not None and test_ric is not None:
        print(f"  Test IC: {test_ic:.4f}, RankIC: {test_ric:.4f}")
    else:
        print("  Test IC: n/a (pool empty)")

    total_factors = len(expressions)
    print(f"\n  Generated {total_factors} factors:")
    fundamental_factor_count = 0
    for i, (expr, w) in enumerate(zip(expressions, weights), 1):
        expr_short = expr if len(expr) <= 60 else expr[:57] + "..."
        fundamentals_in_expr = sorted(token for token in FUNDAMENTAL_TOKENS if token in expr)
        if fundamentals_in_expr:
            fundamental_factor_count += 1
            print(f"    {i}. (w={w:.3f}) {expr_short} | fundamentals: {', '.join(fundamentals_in_expr)}")
        else:
            print(f"    {i}. (w={w:.3f}) {expr_short}")
    if total_factors:
        print(f"  Factors using fundamentals: {fundamental_factor_count}/{total_factors}")
    else:
        print("  Factors using fundamentals: 0/0 (pool empty)")

    output_data = {
        "expressions": expressions,
        "weights": weights,
        "train_ic": float(train_ic),
        "test_ic": None if test_ic is None else float(test_ic),
        "test_ric": None if test_ric is None else float(test_ric),
        "pool_size": int(pool.size),
        "metadata": {
            "ticker_pool": ticker_pool,
            "train_range": [train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")],
            "test_range": [test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d")],
            "train_steps": args.steps,
            "pool_capacity": args.pool_capacity,
            "forward_horizon": args.forward_horizon,
            "timestamp": datetime.now().isoformat()
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Saved to: {output_path}")

    if data_pickle.exists():
        data_pickle.unlink()

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

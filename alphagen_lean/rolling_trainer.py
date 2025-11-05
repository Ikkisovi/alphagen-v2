"""
Rolling window trainer for AlphaGen factors.

Orchestrates the training process across multiple rolling windows.
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import Feature, Ref
from alphagen.local_data import LocalDataConfig, load_local_stock_data
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import FeatureType, StockData

from .rolling_config import RollingConfig
from .window_manager import WindowManager, WindowInfo
from .data_prep import prepare_window_data


class WindowTrainingCallback(BaseCallback):
    """
    Callback for saving pool states and metrics during training of a single window.
    """

    def __init__(self,
                 pool: MseAlphaPool,
                 eval_calculators: Optional[List[Tuple[str, QLibStockDataCalculator]]],
                 window_dir: Path,
                 verbose: int = 1):
        super().__init__(verbose)
        self.pool = pool
        self.eval_calculators = eval_calculators or []
        self.window_dir = window_dir
        self.pool_dir = window_dir / "pool_states"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.metric_log = window_dir / "metrics.log"

    def _on_rollout_end(self) -> None:
        """Called at the end of each PPO rollout."""
        metrics: Dict[str, Dict[str, float]] = {}

        # Evaluate on all segments
        for name, calculator in self.eval_calculators:
            ic, ric = self.pool.test_ensemble(calculator)
            metrics[name] = {"ic": float(ic), "rank_ic": float(ric)}

        metrics["train"] = {
            "ic": float(self.pool.best_ic_ret),
            "rank_ic": float("nan"),
        }

        if self.verbose:
            summary = " | ".join(
                f"{name}: IC={values['ic']:.4f}, RankIC={values['rank_ic']:.4f}"
                for name, values in metrics.items()
            )
            print(f"  [Rollout @ {self.num_timesteps} steps] size={self.pool.size} | {summary}")

        # Log metrics
        log_entry = {
            "timesteps": int(self.num_timesteps),
            "pool_size": int(self.pool.size),
            "metrics": metrics,
            "best_exprs": [str(expr) for expr in self.pool.exprs[: self.pool.size]],
        }
        with self.metric_log.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry) + "\n")

        # Save pool checkpoint
        checkpoint = {
            "timesteps": int(self.num_timesteps),
            "pool": self.pool.to_json_dict(),
            "metrics": metrics,
        }
        ckpt_path = self.pool_dir / f"pool_{self.num_timesteps}.json"
        ckpt_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        # Save model checkpoint
        model_dir = self.window_dir / "checkpoints"
        model_dir.mkdir(exist_ok=True)
        self.model.save(model_dir / f"ppo_{self.num_timesteps}")

    def _on_step(self) -> bool:
        return True


class RollingTrainer:
    """
    Main trainer for rolling window AlphaGen training.

    Orchestrates:
    1. Data preparation for each window
    2. AlphaGen training using RL
    3. Results saving and export
    """

    def __init__(self, config: RollingConfig):
        """
        Initialize rolling trainer.

        Args:
            config: Rolling training configuration
        """
        self.config = config
        self.window_manager = WindowManager(
            first_train_start=config.first_train_start,
            deploy_start=config.deploy_start,
            deploy_end=config.deploy_end,
            train_months=config.train_months,
            test_months=config.test_months,
            step_months=config.step_months
        )

        # Create output directories
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = self.output_dir / "rolling_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        print(f"\n{'='*80}")
        print(f"Rolling Trainer Initialized")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of windows: {self.window_manager.n_windows}")
        print(f"{'='*80}\n")

    def _slice_dataset(self, data: StockData, start: str, end: str) -> StockData:
        """Slice StockData by date range."""
        return data[data.find_date_slice(start, end)]

    def train_single_window(self, window: WindowInfo, data_pickle_path: Path) -> Dict:
        """
        Train AlphaGen on a single window (single-stage training).

        Args:
            window: Window information
            data_pickle_path: Path to prepared data pickle

        Returns:
            Dictionary with training results
        """
        return self._train_stage(
            window=window,
            data_pickle_path=data_pickle_path,
            stage_name="single",
            price_only=False,
            warm_start_exprs=None,
            train_steps=self.config.train_steps
        )

    def train_single_window_dual_stage(self, window: WindowInfo, data_pickle_path: Path) -> Dict:
        """
        Train AlphaGen with two-stage approach:
        1. Price-only stage (OHLCV features)
        2. Fundamental stage (warm-start from stage 1)

        Args:
            window: Window information
            data_pickle_path: Path to prepared data pickle

        Returns:
            Dictionary with fundamental stage training results
        """
        print(f"\n{'='*80}")
        print(f"Dual-Stage Training: Window {window.window_idx}: {window.deploy_month}")
        print(f"{'='*80}\n")

        # Stage 1: Price features only
        print(f"\n{'*'*80}")
        print(f"STAGE 1: Price-Only Features")
        print(f"{'*'*80}\n")

        price_results = self._train_stage(
            window=window,
            data_pickle_path=data_pickle_path,
            stage_name="price",
            price_only=True,
            warm_start_exprs=None,
            train_steps=self.config.price_stage_steps
        )

        # Stage 2: Add fundamentals with warm-start
        print(f"\n{'*'*80}")
        print(f"STAGE 2: Fundamental Features (Warm Start)")
        print(f"{'*'*80}\n")

        fundamental_results = self._train_stage(
            window=window,
            data_pickle_path=data_pickle_path,
            stage_name="fundamental",
            price_only=False,
            warm_start_exprs=price_results['expressions'],
            train_steps=self.config.fundamental_stage_steps
        )

        return fundamental_results

    def _train_stage(
        self,
        window: WindowInfo,
        data_pickle_path: Path,
        stage_name: str,
        price_only: bool,
        warm_start_exprs: Optional[List[str]],
        train_steps: int
    ) -> Dict:
        """
        Train a single stage (price-only or with fundamentals).

        Args:
            window: Window information
            data_pickle_path: Path to prepared data pickle
            stage_name: Name of the stage ("single", "price", or "fundamental")
            price_only: If True, only use OHLCV features
            warm_start_exprs: Optional list of expressions to seed the pool
            train_steps: Number of training steps

        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"Training Window {window.window_idx}: {window.deploy_month} - {stage_name.upper()} Stage")
        print(f"{'='*80}")
        print(f"Train period: {window.train_range[0]} to {window.train_range[1]}")
        print(f"Deploy period: {window.deploy_range[0]} to {window.deploy_range[1]}")
        print(f"Price-only: {price_only}")
        print(f"Warm-start: {warm_start_exprs is not None}")
        print(f"Training steps: {train_steps}")
        print(f"{'='*80}\n")

        # Create window output directory
        if stage_name == "single":
            window_dir = self.output_dir / f"window_{window.deploy_month}"
        else:
            window_dir = self.output_dir / f"window_{window.deploy_month}" / f"{stage_name}_stage"
        window_dir.mkdir(parents=True, exist_ok=True)

        # Save window info
        window_info_path = window_dir / "window_info.json"
        with open(window_info_path, 'w') as f:
            json.dump(window.to_dict(), f, indent=2)

        # Reseed for reproducibility
        reseed_everything(self.config.seed)

        # Load stock data with feature filtering
        print("Loading stock data...")

        # Configure features based on stage
        price_features = [
            FeatureType.OPEN,
            FeatureType.CLOSE,
            FeatureType.HIGH,
            FeatureType.LOW,
            FeatureType.VOLUME,
            FeatureType.VWAP,
        ]
        fund_path: Optional[Path] = None
        if price_only:
            # Price-only stage: use only OHLCV features
            cfg = LocalDataConfig(
                max_backtrack_days=self.config.max_backtrack,
                max_future_days=self.config.max_future,
                features=price_features,
                device=self.config.device,
                fundamental_path=None,  # Disable fundamentals
            )
            print(f"  Using {len(price_features)} price features only")
        else:
            # Full features including fundamentals
            # TEMP: Disable fundamentals if file doesn't exist
            fund_path = self.config.fundamental_path
            if fund_path and not Path(fund_path).exists():
                print(f"  [WARNING] Fundamental file not found: {fund_path}")
                print(f"  Continuing with price features only")
                fund_path = None

            feature_list = list(FeatureType) if fund_path else price_features
            cfg = LocalDataConfig(
                max_backtrack_days=self.config.max_backtrack,
                max_future_days=self.config.max_future,
                device=self.config.device,
                features=feature_list,
                fundamental_path=fund_path,
            )
            if fund_path:
                print(f"  Using all features (including fundamentals)")
            else:
                print(f"  Using price features only (fundamentals disabled)")

        stock = load_local_stock_data(data_pickle_path, config=cfg)
        print(f"Dataset loaded: days={stock.n_days}, stocks={stock.n_stocks}, features={stock.n_features}\n")

        # Define target
        close = Feature(FeatureType.CLOSE)
        target = Ref(close, -self.config.forward_horizon) / close - 1

        # Slice data for training and deployment
        print("Slicing data for train and deploy periods...")
        train = self._slice_dataset(stock, *window.train_range)
        deploy = self._slice_dataset(stock, *window.deploy_range)
        print(f"Train data: {train.n_days} days")
        print(f"Deploy data: {deploy.n_days} days")
        deploy_calculator: Optional[QLibStockDataCalculator] = None
        eval_calculators: List[Tuple[str, QLibStockDataCalculator]] = []

        # Create calculators
        train_calculator = QLibStockDataCalculator(train, target)
        if self.config.evaluate_deploy and deploy.n_days > 0:
            deploy_calculator = QLibStockDataCalculator(deploy, target)
            eval_calculators = [("deploy", deploy_calculator)]
            print("Deploy evaluation enabled\n")
        else:
            reason = "disabled by configuration" if not self.config.evaluate_deploy else "no deploy data available"
            print(f"Skipping deploy evaluation ({reason}).\n")

        # Create alpha pool
        print("Initializing alpha pool...")
        pool = MseAlphaPool(
            capacity=self.config.pool_capacity,
            calculator=train_calculator,
            ic_lower_bound=None,
            l1_alpha=self.config.l1_alpha,
            device=self.config.device,
        )

        # Warm-start: seed pool with previous expressions
        if warm_start_exprs is not None:
            print(f"Seeding pool with {len(warm_start_exprs)} warm-start expressions...")
            from alphagen.data.parser import parse_expression
            for expr_str in warm_start_exprs[:self.config.pool_capacity]:
                try:
                    expr = parse_expression(expr_str)
                    pool.try_new_expr(expr)
                except Exception as e:
                    print(f"  Warning: Failed to parse warm-start expression: {expr_str[:50]}... Error: {e}")
            print(f"  Pool size after warm-start: {pool.size}")

        # Create feature mask for price-only stage
        allowed_features = None
        if price_only or (not price_only and not fund_path):
            # Create mask: only allow price features when fundamentals are unavailable
            allowed_features = price_features
            print(f"  Restricting to {len(allowed_features)} price features")

        # Create RL environment
        env = AlphaEnv(
            pool=pool,
            device=self.config.device,
            allowed_features=allowed_features,
            print_expr=False,
        )

        # Create PPO model
        print("Initializing PPO model...")
        policy_kwargs = dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=self.config.lstm_layers,
                d_model=self.config.lstm_width,
                dropout=self.config.lstm_dropout,
                device=self.config.device,
            ),
        )

        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            gamma=self.config.gamma,
            ent_coef=self.config.ent_coef,
            batch_size=self.config.batch_size,
            n_steps=self.config.n_steps,
            tensorboard_log=str(window_dir / "tensorboard"),
            device=self.config.device,
            verbose=0,
        )

        # Create callback
        callback = WindowTrainingCallback(pool, eval_calculators, window_dir, verbose=1)

        # Train
        print(f"Starting training for {train_steps} steps...\n")
        start_time = time.time()
        model.learn(
            total_timesteps=train_steps,
            callback=callback,
            progress_bar=False
        )
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        # Save final model
        model.save(window_dir / "final_model")

        # Final evaluation
        print("\nFinal evaluation...")
        deploy_ic: Optional[float] = None
        deploy_ric: Optional[float] = None
        if deploy_calculator is not None:
            ic_val, ric_val = pool.test_ensemble(deploy_calculator)
            deploy_ic = float(ic_val)
            deploy_ric = float(ric_val)
        else:
            print("  Deploy evaluation skipped.")

        # Prepare results
        results = {
            'window_idx': window.window_idx,
            'deploy_month': window.deploy_month,
            'train_range': window.train_range,
            'deploy_range': window.deploy_range,
            'train_ic': float(pool.best_ic_ret),
            'deploy_ic': deploy_ic,
            'deploy_ric': deploy_ric,
            'pool_size': int(pool.size),
            'expressions': [str(expr) for expr in pool.exprs[:pool.size]],
            'weights': pool.weights.tolist()[:pool.size],
            'training_time': training_time,
        }

        # Save final report
        final_report_path = window_dir / "final_report.json"
        with open(final_report_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nWindow {window.window_idx} Results:")
        print(f"  Train IC: {results['train_ic']:.4f}")
        deploy_ic_str = f"{results['deploy_ic']:.4f}" if results['deploy_ic'] is not None else "N/A"
        deploy_ric_str = f"{results['deploy_ric']:.4f}" if results['deploy_ric'] is not None else "N/A"
        print(f"  Deploy IC: {deploy_ic_str}")
        print(f"  Deploy RankIC: {deploy_ric_str}")
        print(f"  Pool size: {results['pool_size']}")
        print(f"  Saved to: {window_dir}")

        return results

    def train_all_windows(self, start_window: int = 0, end_window: Optional[int] = None) -> List[Dict]:
        """
        Train all windows sequentially.

        Args:
            start_window: Start from this window index (default: 0)
            end_window: End at this window index (default: all windows)

        Returns:
            List of results dictionaries
        """
        if end_window is None:
            end_window = self.window_manager.n_windows

        print(self.window_manager.summary())
        print(f"\nTraining windows {start_window} to {end_window - 1}...")

        all_results = []

        # First, prepare data for all windows (covering the full date range needed)
        # We need data from first_train_start to deploy_end
        first_window = self.window_manager.get_window(start_window)
        last_window = self.window_manager.get_window(end_window - 1)

        data_start = datetime.strptime(first_window.train_range[0], "%Y-%m-%d")
        data_end = datetime.strptime(last_window.deploy_range[1], "%Y-%m-%d")

        # Add some buffer days for warmup
        from datetime import timedelta
        data_start = data_start - timedelta(days=self.config.max_backtrack)

        data_pickle_path = self.output_dir / "all_windows_data.pkl"

        # Check if data already exists
        if not data_pickle_path.exists():
            print(f"\nPreparing data for all windows...")
            print(f"Date range: {data_start.date()} to {data_end.date()}")
            print(f"Resolution: {self.config.data_resolution}")
            prepare_window_data(
                self.config.data_path,
                self.config.symbols,
                data_start,
                data_end,
                data_pickle_path,
                resolution=self.config.data_resolution
            )
        else:
            print(f"\nUsing existing data: {data_pickle_path}")

        # Train each window
        for window_idx in range(start_window, end_window):
            window = self.window_manager.get_window(window_idx)

            try:
                # Choose training strategy
                if self.config.train_strategy == "dual_stage":
                    results = self.train_single_window_dual_stage(window, data_pickle_path)
                else:
                    results = self.train_single_window(window, data_pickle_path)
                all_results.append(results)
            except Exception as e:
                print(f"\nERROR training window {window_idx}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with next window

        # Save summary
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'n_windows': len(all_results),
                'windows': all_results,
                'config': self.config.to_dict(),
            }, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Rolling Training Complete!")
        print(f"{'='*80}")
        print(f"Trained {len(all_results)} windows")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")

        # Print summary table
        self._print_summary_table(all_results)

        return all_results

    def _print_summary_table(self, results: List[Dict]):
        """Print a summary table of all window results."""
        print("\nSummary Table:")
        print("-" * 100)
        print(f"{'Window':<8} {'Deploy Month':<12} {'Train IC':<12} {'Deploy IC':<12} {'Deploy RankIC':<15} {'Pool Size':<10}")
        print("-" * 100)

        for r in results:
            deploy_ic = r['deploy_ic']
            deploy_ric = r['deploy_ric']
            deploy_ic_str = f"{deploy_ic:>11.4f}" if deploy_ic is not None else f"{'N/A':>11}"
            deploy_ric_str = f"{deploy_ric:>14.4f}" if deploy_ric is not None else f"{'N/A':>14}"
            print(
                f"{r['window_idx']:<8} "
                f"{r['deploy_month']:<12} "
                f"{r['train_ic']:>11.4f} "
                f"{deploy_ic_str} "
                f"{deploy_ric_str} "
                f"{r['pool_size']:>10}"
            )

        print("-" * 100)

        # Calculate averages
        avg_train_ic = np.mean([r['train_ic'] for r in results])
        deploy_ics = [r['deploy_ic'] for r in results if r['deploy_ic'] is not None]
        deploy_rics = [r['deploy_ric'] for r in results if r['deploy_ric'] is not None]
        avg_deploy_ic = np.mean(deploy_ics) if deploy_ics else None
        avg_deploy_ric = np.mean(deploy_rics) if deploy_rics else None

        avg_deploy_ic_str = f"{avg_deploy_ic:>11.4f}" if avg_deploy_ic is not None else f"{'N/A':>11}"
        avg_deploy_ric_str = f"{avg_deploy_ric:>14.4f}" if avg_deploy_ric is not None else f"{'N/A':>14}"

        print(f"{'Average':<8} {'':<12} {avg_train_ic:>11.4f} {avg_deploy_ic_str} {avg_deploy_ric_str}")
        print("-" * 100)


if __name__ == "__main__":
    # Example usage
    config = RollingConfig()
    trainer = RollingTrainer(config)

    # Train first 3 windows as a test
    trainer.train_all_windows(start_window=0, end_window=3)

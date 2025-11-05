"""
Entry point for running AlphaGen's RL-based factor mining against the local
`all_symbols_data.pkl` dataset.

Usage example:

    python -m scripts.run_local --steps 4096 --print-expr
"""

from __future__ import annotations

import argparse
import json
import time
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import numpy as np
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

try:
    import tensorboard  # type: ignore

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "all_symbols_data.pkl",
        help="Path to the local OHLCV pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./out/local_runs"),
        help="Directory where checkpoints and logs are written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--pool-capacity",
        type=int,
        default=10,
        help="Maximum number of alphas kept in the pool.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4096,
        help="Total PPO timesteps. Must be a multiple of n_steps.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="PPO rollout length before each policy update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Minibatch size for PPO optimisation.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor for PPO.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy regularisation coefficient.",
    )
    parser.add_argument(
        "--l1-alpha",
        type=float,
        default=5e-3,
        help="L1 regularisation strength for the linear alpha pool.",
    )
    parser.add_argument(
        "--forward-horizon",
        type=int,
        default=20,
        help="Target forward return horizon (in days).",
    )
    parser.add_argument(
        "--max-backtrack",
        type=int,
        default=120,
        help="Maximum historical look-back (days) allowed in expressions.",
    )
    parser.add_argument(
        "--max-future",
        type=int,
        default=40,
        help="Maximum forward range used for targets (days).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device specifier, e.g. 'cpu' or 'cuda:0'.",
    )
    parser.add_argument(
        "--train-range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        default=("2024-01-02", "2024-12-31"),
        help="Date range used for training.",
    )
    parser.add_argument(
        "--val-range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        default=("2025-01-01", "2025-05-31"),
        help="Date range used for validation IC metrics.",
    )
    parser.add_argument(
        "--test-range",
        type=str,
        nargs=2,
        metavar=("START", "END"),
        default=("2025-06-01", "2025-10-28"),
        help="Date range used for out-of-sample evaluation.",
    )
    parser.add_argument(
        "--print-expr",
        action="store_true",
        help="Stream each accepted expression to stdout for inspection.",
    )
    parser.add_argument(
        "--lstm-width",
        type=int,
        default=128,
        help="Hidden size of the shared LSTM policy backbone.",
    )
    parser.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers in the shared backbone.",
    )
    parser.add_argument(
        "--lstm-dropout",
        type=float,
        default=0.1,
        help="Dropout applied between LSTM layers.",
    )
    return parser.parse_args()


def _slice_dataset(data: StockData, start: str, end: str) -> StockData:
    return data[data.find_date_slice(start, end)]


class EvaluationCallback(BaseCallback):
    """
    Save pool states and report IC metrics at the end of every PPO rollout.
    """

    def __init__(
        self,
        pool: MseAlphaPool,
        eval_calculators: Sequence[Tuple[str, QLibStockDataCalculator]],
        run_dir: Path,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.pool = pool
        self.eval_calculators = eval_calculators
        self.run_dir = run_dir
        self.pool_dir = run_dir / "pool_states"
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.metric_log = run_dir / "metrics.log"

    def _on_rollout_end(self) -> None:
        metrics: Dict[str, Dict[str, float]] = {}
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
            print(f"[Rollout @ {self.num_timesteps} steps] size={self.pool.size} | {summary}")

        log_entry = {
            "timesteps": int(self.num_timesteps),
            "pool_size": int(self.pool.size),
            "metrics": metrics,
            "best_exprs": [str(expr) for expr in self.pool.exprs[: self.pool.size]],
        }
        with self.metric_log.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log_entry) + "\n")

        checkpoint = {
            "timesteps": int(self.num_timesteps),
            "pool": self.pool.to_json_dict(),
            "metrics": metrics,
        }
        ckpt_path = self.pool_dir / f"pool_{self.num_timesteps}.json"
        ckpt_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        model_dir = self.run_dir / "checkpoints"
        model_dir.mkdir(exist_ok=True)
        self.model.save(model_dir / f"ppo_{self.num_timesteps}")  # type: ignore[arg-type]

    def _on_step(self) -> bool:
        return True


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    reseed_everything(args.seed)

    output_root = args.output_dir.expanduser().resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Outputs will be stored in: {run_dir}")

    cfg = LocalDataConfig(
        max_backtrack_days=args.max_backtrack,
        max_future_days=args.max_future,
        device=device,
    )
    if cfg.max_future_days < args.forward_horizon:
        raise ValueError(
            f"max_future ({cfg.max_future_days}) must be >= forward_horizon ({args.forward_horizon})"
        )

    print("[INFO] Loading local dataset ...")
    stock = load_local_stock_data(args.data_path, config=cfg)
    print(
        f"[INFO] Dataset loaded: days={stock.n_days}, stocks={stock.n_stocks}, "
        f"features={stock.n_features}"
    )

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -args.forward_horizon) / close - 1

    train = _slice_dataset(stock, *args.train_range)
    calculators: List[Tuple[str, QLibStockDataCalculator]] = []
    for name, bounds in (
        ("val", args.val_range),
        ("test", args.test_range),
    ):
        try:
            segment = _slice_dataset(stock, *bounds)
            calculators.append((name, QLibStockDataCalculator(segment, target)))
        except ValueError as exc:
            print(f"[WARN] Skipping {name} segment: {exc}")

    train_calculator = QLibStockDataCalculator(train, target)
    pool = MseAlphaPool(
        capacity=args.pool_capacity,
        calculator=train_calculator,
        ic_lower_bound=None,
        l1_alpha=args.l1_alpha,
        device=device,
    )

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=args.print_expr,
    )

    policy_kwargs = dict(
        features_extractor_class=LSTMSharedNet,
        features_extractor_kwargs=dict(
            n_layers=args.lstm_layers,
            d_model=args.lstm_width,
            dropout=args.lstm_dropout,
            device=device,
        ),
    )

    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_log = str(tensorboard_dir) if HAS_TENSORBOARD else None
    if HAS_TENSORBOARD:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    else:
        print("[INFO] TensorBoard not installed; skipping tensorboard logging.")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        tensorboard_log=tensorboard_log,
        device=device,
        verbose=1,
    )

    callback = EvaluationCallback(pool, calculators, run_dir, verbose=1)

    print(
        f"[INFO] Starting training for {args.steps} timesteps "
        f"(n_steps={args.n_steps}, batch_size={args.batch_size})"
    )
    model.learn(total_timesteps=args.steps, callback=callback, progress_bar=False)
    model.save(run_dir / "final_model")

    summary: Dict[str, Tuple[float, float]] = {}
    for name, calculator in calculators:
        summary[name] = pool.test_ensemble(calculator)
    final_report = {
        "train_ic": float(pool.best_ic_ret),
        "segments": {
            name: {"ic": float(ic), "rank_ic": float(ric)}
            for name, (ic, ric) in summary.items()
        },
        "expressions": pool.to_json_dict(),
    }
    (run_dir / "final_report.json").write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print("[INFO] Final ensemble metrics:")
    for name, (ic, ric) in summary.items():
        print(f"  - {name:5s} -> IC: {ic:.4f}, RankIC: {ric:.4f}")

    # Save factor values for each segment
    print("[INFO] Computing and saving factor values...")
    for name, calculator in calculators:
        try:
            with torch.no_grad():
                factor_values = calculator.make_ensemble_alpha(
                    pool.exprs[:pool.size],
                    pool.weights
                ).cpu().numpy()

            # Save as pickle
            factor_path = run_dir / f"factor_values_{name}.pkl"
            with open(factor_path, "wb") as f:
                pickle.dump({
                    "factor_values": factor_values,
                    "expressions": [str(expr) for expr in pool.exprs[:pool.size]],
                    "weights": pool.weights.tolist(),
                    "shape": factor_values.shape,
                    "description": f"Factor values for {name} segment (days x stocks)"
                }, f)

            # Also save as CSV for easy inspection (first 100 stocks)
            csv_path = run_dir / f"factor_values_{name}_preview.csv"
            df = pd.DataFrame(
                factor_values[:, :min(100, factor_values.shape[1])],
                columns=[f"stock_{i}" for i in range(min(100, factor_values.shape[1]))]
            )
            df.to_csv(csv_path, index=True)

            print(f"  - Saved {name} factor values: {factor_values.shape} -> {factor_path}")
        except Exception as e:
            print(f"  - Warning: Failed to save {name} factor values: {e}")


if __name__ == "__main__":
    main()

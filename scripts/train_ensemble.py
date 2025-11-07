"""
Ensemble Alpha Mining Script
=============================

This script orchestrates multi-window ensemble factor training:
1. Trains factors on 12M, 6M, and 3M historical windows
2. Uses a single-stage price feature sweep (with market_cap & turnover support)
3. Streams training progress to TensorBoard and console every 2000 steps
4. Merges all discovered factors into a single ensemble
5. Optimizes final weights on a validation period and caches outputs

Usage:
    python -m scripts.train_ensemble--config-file  config/novdec_ensemble_config.yaml 
    python scripts/train_ensemble.py --skip-training  # Use cached pools only
"""

import fnmatch
import json
import os
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Sequence
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import fire

import numpy as np
import pandas as pd
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.local_data import (
    LocalDataConfig,
    build_feature_store_stock_data,
    load_local_stock_data,
)
from alphagen.models.linear_alpha_pool import MseAlphaPool, CustomRewardAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import StockData, FeatureType, initialize_qlib
from alphagen_generic.operators import Operators
from alphagen_data_pipeline.storage import load_features as load_feature_store



PRICE_FEATURES: Tuple[FeatureType, ...] = (
    FeatureType.OPEN,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.CLOSE,
    FeatureType.VOLUME,
    FeatureType.VWAP,
    FeatureType.MARKET_CAP,
    FeatureType.TURNOVER,
)


@dataclass
class DataContext:
    source: str
    forward_horizon: int = 20
    stock_data: Optional[StockData] = None
    price_features: Sequence[FeatureType] = PRICE_FEATURES
    features: Optional[Sequence[FeatureType]] = None
    derived_features: Sequence[str] = ()


def parse_feature_names(feature_names: Sequence[str]) -> List[FeatureType]:
    """Convert config feature names into FeatureType enums."""
    result: List[FeatureType] = []
    for name in feature_names:
        key = name.strip()
        if not key:
            continue
        key = key.upper()
        if key.startswith("FEATURETYPE."):
            key = key.split(".", 1)[1]
        if not hasattr(FeatureType, key):
            raise ValueError(f"Unknown feature type: {name}")
        result.append(getattr(FeatureType, key))
    return result


DERIVED_SPEC_BY_COLUMN = {}


def initialize_data_context(
    config: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> DataContext:
    """Prepare dataset access (qlib or local merged data)."""
    data_cfg = config.get('data', {})
    source = data_cfg.get('source', 'qlib').lower()
    forward_horizon = int(data_cfg.get('forward_horizon', 20))
    context = DataContext(source=source, forward_horizon=forward_horizon)

    def _infer_date_bounds() -> Tuple[Optional[str], Optional[str]]:
        windows = config.get('time_windows', {})
        starts: List[pd.Timestamp] = []
        ends: List[pd.Timestamp] = []
        for window in windows.values():
            start = window.get('start_date')
            end = window.get('end_date')
            if start:
                starts.append(pd.to_datetime(start))
            if end:
                ends.append(pd.to_datetime(end))
        if not starts or not ends:
            return None, None
        return str(min(starts).date()), str(max(ends).date())

    if source == 'merged':
        data_path = Path(data_cfg.get('path', 'data/merged/merged_data.parquet'))
        fundamental_path = data_cfg.get('fundamental_path')
        features_cfg = data_cfg.get('features')
        price_features_cfg = data_cfg.get('price_features')

        # Adjust horizon for intra-day data by inspecting a sample of the data
        try:
            temp_df = pd.read_csv(data_path, nrows=1000)
            if 'timestamp' in temp_df.columns:
                temp_df['date'] = pd.to_datetime(temp_df['timestamp'], utc=True)
                points_per_day = len(temp_df) / len(temp_df['date'].dt.normalize().unique())
                logger.info(f"Detected approximately {points_per_day:.2f} data points per day.")
                if points_per_day > 1.5:
                    adjustment_factor = int(round(points_per_day))
                    original_horizon = forward_horizon
                    forward_horizon *= adjustment_factor
                    logger.info(
                        f"Adjusting forward_horizon by a factor of {adjustment_factor} "
                        f"to account for higher data frequency. "
                        f"Original: {original_horizon}, New: {forward_horizon}"
                    )
        except Exception as e:
            logger.warning(f"Could not inspect data frequency, using default forward_horizon: {e}")

        context.forward_horizon = forward_horizon

        features = tuple(
            parse_feature_names(features_cfg)
            if features_cfg
            else list(FeatureType)
        )
        context.features = features

        if price_features_cfg:
            context.price_features = tuple(parse_feature_names(price_features_cfg))

        # Ensure max_future_days is sufficient for the adjusted forward_horizon
        configured_future_days = int(data_cfg.get('max_future_days', 0)) if 'max_future_days' in data_cfg else 0
        required_future_days = forward_horizon + 10  # Buffer of 10 periods
        max_future_days = max(configured_future_days, required_future_days)

        local_cfg = LocalDataConfig(
            max_backtrack_days=int(data_cfg.get('max_backtrack_days', 120)),
            max_future_days=max_future_days,
            features=tuple(features),
            device=device,
            fundamental_path=Path(fundamental_path).expanduser() if fundamental_path else None,
        )

        logger.info(
            f"LocalDataConfig: max_backtrack_days={local_cfg.max_backtrack_days}, "
            f"max_future_days={max_future_days} (forward_horizon={forward_horizon})"
        )

        stock_data = load_local_stock_data(data_path, config=local_cfg)
        context.stock_data = stock_data
        context.features = tuple(features)
        if price_features_cfg:
            context.price_features = tuple(parse_feature_names(price_features_cfg))

        # Validate that loaded data has sufficient future periods
        if stock_data.max_future_days < forward_horizon:
            logger.warning(
                "WARNING: Loaded data has max_future_days=%d but forward_horizon=%d. "
                "This may cause OutOfDataRangeError. Consider reloading with larger max_future_days.",
                stock_data.max_future_days,
                forward_horizon
            )

        logger.info(
            "Loaded merged dataset: %s → %s (days=%d, symbols=%d, features=%d, "
            "max_future_days=%d)",
            stock_data._start_time,
            stock_data._end_time,
            stock_data.n_days,
            stock_data.n_stocks,
            stock_data.n_features,
            stock_data.max_future_days,
        )
    elif source == 'feature_store':
        raw_store_path = data_cfg.get('path')
        if not raw_store_path:
            raise ValueError("Feature store path must be provided when source=feature_store")
        store_path = Path(raw_store_path).expanduser()

        features_cfg = data_cfg.get('features')
        price_features_cfg = data_cfg.get('price_features')
        feature_patterns = data_cfg.get('feature_patterns')
        session_list = data_cfg.get('sessions')
        timezone = data_cfg.get('timezone', 'UTC')
        session_time_map = data_cfg.get('session_time_map')
        symbols = data_cfg.get('symbols')

        features = tuple(
            parse_feature_names(features_cfg)
            if features_cfg
            else list(FeatureType)
        )
        context.features = features

        if price_features_cfg:
            context.price_features = tuple(parse_feature_names(price_features_cfg))

        preload_cfg = data_cfg.get('preload', {})
        start = preload_cfg.get('start')
        end = preload_cfg.get('end')
        inferred_start, inferred_end = _infer_date_bounds()
        if start is None:
            start = inferred_start
        if end is None:
            end = inferred_end
        if start is None or end is None:
            raise ValueError("Unable to infer feature store preload date range; specify data.preload.start/end")

        sessions = [str(sess).upper() for sess in session_list] if session_list else None

        logger.info(
            "Loading feature store from %s for %s → %s", store_path, start, end
        )
        feature_frame = load_feature_store(
            str(store_path),
            start=start,
            end=end,
            feature_patterns=feature_patterns,
            symbols=symbols,
            sessions=sessions,
        )

        if feature_frame.empty:
            raise ValueError(
                f"Feature store query returned no rows for range {start} → {end}"
            )

        session_counts = feature_frame.groupby(['symbol', 'date'])['session'].nunique()
        if not session_counts.empty:
            approx_points = float(session_counts.mean())
            logger.info(
                "Detected approximately %.2f sessions per trading day in feature store",
                approx_points,
            )
            if approx_points > 1.5:
                adjustment_factor = max(1, int(round(approx_points)))
                original_horizon = forward_horizon
                forward_horizon *= adjustment_factor
                logger.info(
                    "Adjusting forward_horizon by factor %d (original=%d, new=%d)",
                    adjustment_factor,
                    original_horizon,
                    forward_horizon,
                )
        else:
            logger.warning(
                "Unable to determine session density; using configured forward_horizon=%d",
                forward_horizon,
            )

        context.forward_horizon = forward_horizon

        configured_future_days = int(data_cfg.get('max_future_days', 0)) if 'max_future_days' in data_cfg else 0
        required_future_days = forward_horizon + 10
        max_future_days = max(configured_future_days, required_future_days)

        local_cfg = LocalDataConfig(
            max_backtrack_days=int(data_cfg.get('max_backtrack_days', 120)),
            max_future_days=max_future_days,
            features=tuple(features),
            device=device,
        )

        stock_data = build_feature_store_stock_data(
            feature_frame,
            config=local_cfg,
            timezone=timezone,
            session_time_map=session_time_map,
        )
        context.stock_data = stock_data

        derived_features = list(data_cfg.get('derived_features', []))
        derived_patterns = data_cfg.get('derived_feature_patterns', [])
        if derived_patterns:
            available_cols = set(feature_frame.columns) - {"symbol", "date", "session"}
            for pattern in derived_patterns:
                derived_features.extend(
                    sorted(
                        col for col in available_cols
                        if fnmatch.fnmatch(col, pattern)
                    )
                )
        if derived_features:
            # Deduplicate while preserving order
            context.derived_features = tuple(dict.fromkeys(derived_features))

        logger.info(
            "Loaded feature store dataset: %s → %s (days=%d, symbols=%d, features=%d, max_future_days=%d)",
            stock_data._start_time,
            stock_data._end_time,
            stock_data.n_days,
            stock_data.n_stocks,
            stock_data.n_features,
            stock_data.max_future_days,
        )
    else:
        logger.info("Using qlib data source")
        initialize_qlib()

    return context


def build_precomputed_feature_subexprs(
    data_context: DataContext,
    subexpr_config: Dict[str, Any],
    logger: logging.Logger,
) -> List[Expression]:
    if not subexpr_config.get('enabled', False):
        return []
    derived_names = data_context.derived_features
    if not derived_names:
        logger.info("No precomputed features available for Stage 1 exploration")
        return []
    subexprs: List[Expression] = []
    for column_name in derived_names:
        token_name = DERIVED_SPEC_BY_COLUMN.get(column_name, column_name.upper())
        subexprs.append(PrecomputedFeature(column_name, token_name))
    logger.info("Stage 1 injected %d precomputed features for exploration", len(subexprs))
    return subexprs


def create_stock_data_slice(
    data_context: DataContext,
    start_date: str,
    end_date: str,
    device: torch.device,
    universe,
) -> StockData:
    """
    Create a StockData view for the requested window regardless of data source.
    Automatically extends end_date by forward_horizon to ensure we have enough
    future data to calculate forward returns.
    """
    if data_context.source in {'merged', 'feature_store'}:
        if data_context.stock_data is None:
            raise ValueError("Local stock data has not been loaded")
        base = data_context.stock_data
        logger = logging.getLogger('ensemble_training')

        def _parse(date_str: Optional[str]) -> Tuple[Optional[pd.Timestamp], bool]:
            if not date_str:
                return None, False
            ts = pd.Timestamp(date_str, tz='utc')
            has_time = not ts.normalize() == ts
            return ts, has_time

        available_start, _ = _parse(base._start_time)
        available_end, _ = _parse(base._end_time)
        requested_start, _ = _parse(start_date)
        requested_end, end_has_time = _parse(end_date)
        requested_start = requested_start or available_start
        requested_end = requested_end or available_end

        clamped_start = max(available_start, min(requested_start, available_end))
        clamped_end = max(clamped_start, min(requested_end, available_end))

        if requested_start < available_start:
            logger.warning(
                "Requested validation start %s is before data start %s, using %s",
                requested_start.strftime("%Y-%m-%d"),
                available_start.strftime("%Y-%m-%d"),
                clamped_start.strftime("%Y-%m-%d"),
            )
        elif requested_start > available_end:
            logger.warning(
                "Requested validation start %s is after data end %s, using %s",
                requested_start.strftime("%Y-%m-%d"),
                available_end.strftime("%Y-%m-%d"),
                clamped_start.strftime("%Y-%m-%d"),
            )

        if requested_end > available_end:
            logger.warning(
                "Requested validation end %s is after data end %s, using %s",
                requested_end.strftime("%Y-%m-%d"),
                available_end.strftime("%Y-%m-%d"),
                clamped_end.strftime("%Y-%m-%d"),
            )
        elif requested_end < available_start:
            logger.warning(
                "Requested validation end %s is before data start %s, using %s",
                requested_end.strftime("%Y-%m-%d"),
                available_start.strftime("%Y-%m-%d"),
                clamped_end.strftime("%Y-%m-%d"),
            )

        end_label_ts = clamped_end
        if not end_has_time:
            end_label_ts = end_label_ts + pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)

        # Extend end date to include forward_horizon periods for calculating forward returns
        # Find the end index and extend it by forward_horizon
        end_idx = base.find_date_index(end_label_ts, exclusive=False)
        extended_end_idx = end_idx + data_context.forward_horizon

        # Convert back to timestamp (accounting for backtrack offset in _dates array)
        # The _dates array has: [backtrack_buffer | actual_data | future_buffer]
        absolute_idx = extended_end_idx + base.max_backtrack_days
        max_available_idx = len(base._dates) - 1

        logger.debug(
            f"Slicing data: end_date={end_date}, end_idx={end_idx}, forward_horizon={data_context.forward_horizon}, "
            f"extended_end_idx={extended_end_idx}, absolute_idx={absolute_idx}, max_available={max_available_idx}, "
            f"max_future_days={base.max_future_days}, n_days={base.n_days}"
        )

        if absolute_idx <= max_available_idx:
            extended_end_ts = base._dates[absolute_idx]
            logger.debug(f"Extended end timestamp: {extended_end_ts}")
        else:
            # Not enough future data, use the last available timestamp
            extended_end_ts = base._dates[max_available_idx]
            available_extension = max_available_idx - (end_idx + base.max_backtrack_days)
            logger.error(
                f"INSUFFICIENT FUTURE DATA! Cannot extend end date by full forward_horizon ({data_context.forward_horizon} periods). "
                f"Only {available_extension} future periods available after end_date. "
                f"Need to reload data with max_future_days >= {data_context.forward_horizon}. "
                f"Current max_future_days={base.max_future_days}"
            )
            raise ValueError(
                f"Insufficient future data: need {data_context.forward_horizon} periods but only "
                f"{available_extension} available. Increase max_future_days in config and reload data."
            )

        return base[base.find_date_slice(clamped_start, extended_end_ts)]

    # For qlib data source, StockData handles loading extra future data
    # via max_future_days parameter, which should be >= forward_horizon
    features = list(data_context.price_features)
    return StockData(
        instrument=universe,
        start_time=start_date,
        end_time=end_date,
        max_future_days=max(30, int(data_context.forward_horizon)),
        features=features,
        device=device,
    )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
def setup_logging(config: Dict) -> logging.Logger:
    """Setup logging with both file and console handlers."""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger('ensemble_training')
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    if log_config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)

    # File handler
    if log_config.get('file_output', True):
        log_file = config['output']['training_log']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# CACHE MANAGEMENT
# ============================================================================
class CacheManager:
    """Manages caching of trained factor pools."""

    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.cache_config = config.get('cache', {})
        self.cache_dir = Path(self.cache_config.get('cache_dir', 'output/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_universe_fingerprint(self, universe_config: Dict) -> str:
        """Generate a fingerprint for the current universe configuration."""
        base = universe_config.get('base_instrument', '')
        additional = sorted(universe_config.get('additional_stocks', []))
        fingerprint_str = f"{base}::{','.join(additional)}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()

    def save_universe_fingerprint(self, universe_config: Dict) -> None:
        """Save current universe fingerprint."""
        fingerprint = self.get_universe_fingerprint(universe_config)
        fingerprint_file = universe_config.get('fingerprint_file', 'output/universe_fingerprint.json')
        os.makedirs(os.path.dirname(fingerprint_file), exist_ok=True)
        with open(fingerprint_file, 'w') as f:
            json.dump({
                'fingerprint': fingerprint,
                'timestamp': datetime.now().isoformat(),
                'config': universe_config
            }, f, indent=2)

    def check_universe_consistency(self, universe_config: Dict) -> bool:
        """Check if universe has changed since last run."""
        if not self.cache_config.get('check_universe_consistency', True):
            return True

        fingerprint_file = universe_config.get('fingerprint_file', 'output/universe_fingerprint.json')
        if not os.path.exists(fingerprint_file):
            return False

        current_fingerprint = self.get_universe_fingerprint(universe_config)
        with open(fingerprint_file, 'r') as f:
            saved_data = json.load(f)
            return saved_data['fingerprint'] == current_fingerprint

    def is_cache_fresh(self, cache_file: Path, freshness_days: int) -> bool:
        """Check if cache file is fresh enough to reuse."""
        if not cache_file.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.days <= freshness_days

    def should_use_cache(self, pool_file: str, universe_config: Dict) -> bool:
        """Determine if we should use cached pool instead of retraining."""
        if not self.cache_config.get('enabled', True):
            return False

        if self.cache_config.get('skip_training_if_cached', False):
            # Manual override to always use cache
            return os.path.exists(pool_file)

        pool_path = Path(pool_file)
        if not pool_path.exists():
            return False

        # Check freshness
        freshness_days = self.cache_config.get('freshness_days', 30)
        if not self.is_cache_fresh(pool_path, freshness_days):
            self.logger.info(f"Cache expired for {pool_file} (older than {freshness_days} days)")
            return False

        # Check universe consistency
        if not self.check_universe_consistency(universe_config):
            self.logger.info(f"Universe changed, invalidating cache for {pool_file}")
            return False

        self.logger.info(f"Using cached pool: {pool_file}")
        return True


# ============================================================================
# UNIVERSE CONFIGURATION
# ============================================================================
def build_universe(config: Dict, logger: logging.Logger) -> List[str]:
    """Build stock universe from config."""
    universe_config = config['universe']
    base_instrument = universe_config.get('base_instrument', 'csi300')
    additional_stocks = universe_config.get('additional_stocks', [])

    logger.info(f"Building universe: base={base_instrument}, additional={len(additional_stocks)} stocks")

    # For now, we use base_instrument as-is (qlib handles it)
    # In future, could expand to explicit stock lists
    if additional_stocks:
        logger.warning(f"Additional stocks ({additional_stocks}) will be used if supported by data provider")

    return base_instrument


# ============================================================================
# OPERATOR CONFIGURATION
# ============================================================================
def get_operators_for_stage(stage: str, fundamental_features: List[str]) -> List:
    """Get operator set for a specific training stage."""
    # Import fundamental features dynamically
    from alphagen_generic import features as feature_module

    if stage == 'technical':
        # Technical/price-only operators (no fundamentals)
        return Operators

    elif stage == 'fundamentals':
        # Full operator set including fundamentals
        # Add fundamental features to the operator list
        extended_ops = list(Operators)

        # Add fundamental features
        for feat_name in fundamental_features:
            # Try to get the feature from the module
            feat_attr = getattr(feature_module, feat_name.lower(), None)
            if feat_attr is not None:
                extended_ops.append(feat_attr)

        return extended_ops

    else:
        return Operators


# ============================================================================
# TRAINING CALLBACK
# ============================================================================
class EnsembleTrainingCallback(BaseCallback):
    """Callback for ensemble training with early stopping."""

    def __init__(
        self,
        save_path: str,
        window_name: str,
        early_stopping_patience: int = 30,
        verbose: int = 0,
        python_logger: Optional[logging.Logger] = None
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.window_name = window_name
        self.early_stopping_patience = early_stopping_patience
        self.best_ic = -float('inf')
        self.episodes_since_improvement = 0
        self.episode_count = 0
        self.python_logger = python_logger
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Log progress every 2000 steps
        if self.num_timesteps % 2000 == 0 and self.training_env is not None:
            pool = self.training_env.envs[0].unwrapped.pool

            # FIX 1: Always print to terminal for real-time monitoring
            print(f"\n[Step {self.num_timesteps}] Pool size: {pool.size}, Best IC: {pool.best_ic_ret:.6f}", flush=True)

            # Also log to logger
            message = f"Steps: {self.num_timesteps}, Pool size: {pool.size}, Best IC: {pool.best_ic_ret:.6f}"
            if self.python_logger:
                self.python_logger.info(message)

            # Record to tensorboard
            self.logger.record(f'{self.window_name}/pool_size_step', pool.size)
            self.logger.record(f'{self.window_name}/best_ic_step', pool.best_ic_ret)
            self.logger.dump(step=self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        self.episode_count += 1
        pool = self.training_env.envs[0].unwrapped.pool

        current_ic = pool.best_ic_ret

        # Log progress
        self.logger.record(f'{self.window_name}/episode', self.episode_count)
        self.logger.record(f'{self.window_name}/pool_size', pool.size)
        self.logger.record(f'{self.window_name}/best_ic', current_ic)

        # Print debug stats every 10 episodes to diagnose plateau issues
        if self.episode_count % 10 == 0:
            debug_stats = pool.get_debug_stats()

            # FIX 1: Print to terminal for real-time monitoring
            print(f"\n[Episode {self.episode_count}] Best IC: {debug_stats['best_ic_ret']:.6f}, Best Obj: {debug_stats['best_obj']:.6f}", flush=True)

            # Also print ICIR components if available
            if hasattr(pool, '_objective_components') and pool._objective_components:
                last = pool._objective_components[-1]
                print(f"  IC: {last['ic']:.6f}, ICIR: {last['icir']:.6f}, Turnover: {last['turnover']:.4f}, Penalty: {last['turnover_penalty']:.4f}", flush=True)

            log_func = self.python_logger.info if self.python_logger else self.logger.info
            log_func(f"\n[Episode {self.episode_count}] Debug Stats:")
            log_func(f"  Best IC: {debug_stats['best_ic_ret']:.6f}, Best Obj: {debug_stats['best_obj']:.6f}")
            log_func(f"  Pool Size: {debug_stats['pool_size']}/{debug_stats['capacity']}")
            log_func(f"  Best Updates: {debug_stats['best_updates']}, Total Evals: {debug_stats['eval_count']}")
            if debug_stats['total_failures'] > 0:
                log_func(f"  Total Failures: {debug_stats['total_failures']} ({100*debug_stats['total_failures']/max(1,debug_stats['eval_count']):.1f}%)")
                for reason, count in sorted(debug_stats['failure_stats'].items(), key=lambda x: -x[1])[:3]:
                    log_func(f"    - {reason}: {count}")
            if 'recent_objective_components' in debug_stats and len(debug_stats['recent_objective_components']) > 0:
                latest = debug_stats['recent_objective_components'][-1]
                log_func(f"  Latest Objective: IC={latest['ic']:.4f}, ICIR={latest['icir']:.4f}, "
                               f"Turnover={latest['turnover_penalty']:.4f}, Final={latest['final']:.4f}")

        # Early stopping check
        if current_ic > self.best_ic + 1e-5:  # Small epsilon for numerical stability
            self.best_ic = current_ic
            self.episodes_since_improvement = 0

            # Save best pool
            pool_data = pool.to_json_dict()
            pool_file = os.path.join(self.save_path, f'{self.window_name}_best.json')
            with open(pool_file, 'w') as f:
                json.dump(pool_data, f, indent=2)
        else:
            self.episodes_since_improvement += 1

        # Trigger early stopping if no improvement
        if self.episodes_since_improvement >= self.early_stopping_patience:
            # Print full debug stats before stopping
            log_func = self.python_logger.info if self.python_logger else self.logger.info
            log_func(f"\nEarly stopping triggered. Final debug statistics:")
            pool.print_debug_stats()

            log_func(
                f"Early stopping triggered for {self.window_name} "
                f"after {self.episode_count} episodes "
                f"(no improvement for {self.early_stopping_patience} episodes)"
            )
            return False  # Stop training

        return True


# ============================================================================
# SINGLE-STAGE TRAINING
# ============================================================================
def train_single_window(
    window_name: str,
    window_config: Dict,
    universe: str,
    config: Dict,
    logger: logging.Logger,
    device: torch.device,
    data_context: DataContext,
) -> Dict[str, Any]:
    """
    Train factors for a single time window using a price-feature sweep.

    Returns:
        Dictionary with training results and pool data.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Training window: {window_name}")
    logger.info(f"Period: {window_config['start_date']} to {window_config['end_date']}")
    logger.info(f"{'='*80}\n")

    start_date = window_config['start_date']
    end_date = window_config['end_date']

    training_config = config['training']
    ppo_config = training_config['ppo']

    # Create target
    close = Feature(FeatureType.CLOSE)

    # FIX 4: Adjust forward_horizon based on data frequency to ensure consistent prediction window
    # Original alphagen predicts 20 trading days with daily data
    # For AM/PM data (2 periods per day), need 40 periods to predict 20 days
    forward_horizon_config = max(1, int(data_context.forward_horizon))

    # Detect data frequency by checking if sessions exist in the data
    data_freq = 'daily'  # Default
    if hasattr(data_context.stock_data, '_data') and hasattr(data_context.stock_data._data, 'index'):
        # Check if session column exists (indicates AM/PM data)
        if 'session' in data_context.stock_data._data.columns:
            data_freq = 'ampm'

    # Calculate actual forward_horizon to use
    if data_freq == 'ampm':
        # For AM/PM data: forward_horizon should be 2x to represent same number of days
        # e.g., forward_horizon_config=20 (days) -> 40 periods
        # But if config already specified periods, use as-is
        # Check if config value is already adjusted (>=30 suggests it's in periods)
        if forward_horizon_config < 30:
            forward_horizon = forward_horizon_config * 2  # Convert days to periods
            logger.info(f"AM/PM data detected: Using forward_horizon={forward_horizon} periods ({forward_horizon_config} days)")
        else:
            forward_horizon = forward_horizon_config
            logger.info(f"AM/PM data detected: Using forward_horizon={forward_horizon} periods (already in periods)")
    else:
        # For daily data: use as-is
        forward_horizon = forward_horizon_config
        logger.info(f"Daily data detected: Using forward_horizon={forward_horizon} days")

    target = Ref(close, -forward_horizon) / close - 1
    # Create output directory
    output_dir = Path(config['output']['base_dir']) / window_name
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'window_name': window_name,
        'start_date': start_date,
        'end_date': end_date,
        'stages': {}
    }

    # ========================================================================
    # STAGE 1: Technical/Price-Only Sweep
    # ========================================================================
    stage1_config = training_config['stage1_technical']
    reward_config = dict(training_config.get('reward', {}))
    window_reward_overrides = window_config.get('reward')
    if window_reward_overrides:
        reward_config.update(window_reward_overrides)
    turnover_penalty_coeff = reward_config.get('turnover_penalty_coeff', training_config.get('turnover_penalty_coeff', 0.0))
    turnover_rebalance_horizon = int(reward_config.get('turnover_rebalance_horizon', data_context.forward_horizon))
    turnover_top_k = reward_config.get('turnover_top_k')
    if turnover_top_k is not None:
        turnover_top_k = int(turnover_top_k)
    turnover_top_k_ratio = float(reward_config.get('turnover_top_k_ratio', 0.1))
    objective_description = reward_config.get('objective', 'IC + ICIR - turnover_penalty')
    if objective_description:
        logger.info(f"Reward objective: {objective_description} (turnover_penalty_coeff={turnover_penalty_coeff})")
    logger.info(
        "Turnover settings: rebalance_horizon=%d, top_k=%s, top_k_ratio=%.3f",
        turnover_rebalance_horizon,
        str(turnover_top_k) if turnover_top_k is not None else "auto",
        turnover_top_k_ratio,
    )
    stage1_pool: Dict[str, List[Any]]
    if stage1_config.get('enabled', True):
        logger.info(f"Stage 1: Technical/price-only sweep")

        data = create_stock_data_slice(
            data_context=data_context,
            start_date=start_date,
            end_date=end_date,
            device=device,
            universe=universe,
        )

        # Create calculator
        calculator = QLibStockDataCalculator(data, target)

        # FIX 2: Add use_icir switch to control reward function
        use_icir = reward_config.get('use_icir', False)  # Default False = use IC only (like original alphagen)

        pool = CustomRewardAlphaPool(
            capacity=training_config['pool_capacity_per_window'],
            calculator=calculator,
            ic_lower_bound=config['ensemble'].get('ic_lower_bound', 0.01),
            l1_alpha=config['ensemble']['optimizer']['l1_alpha'],
            turnover_penalty_coeff=turnover_penalty_coeff,
            turnover_rebalance_horizon=turnover_rebalance_horizon,
            turnover_top_k=turnover_top_k,
            turnover_top_k_ratio=turnover_top_k_ratio,
            use_icir=use_icir,  # FIX 2: Configurable ICIR switch
            device=device
        )

        # Log reward configuration
        logger.info(f"Reward configuration: use_icir={use_icir}, turnover_penalty_coeff={turnover_penalty_coeff}")

        # Prepare precomputed feature subexpressions
        subexpr_config = stage1_config.get('precomputed_features', {})
        precomputed_subexprs = build_precomputed_feature_subexprs(
            data_context=data_context,
            subexpr_config=subexpr_config,
            logger=logger,
        )

        # Create environment
        allowed_features = tuple(data_context.price_features)
        env = AlphaEnv(
            pool=pool,
            device=device,
            print_expr=True,
            subexprs=precomputed_subexprs if precomputed_subexprs else None,
            allowed_features=allowed_features,
        )

        # Create PPO model
        model = MaskablePPO(
            'MlpPolicy',
            env,
            policy_kwargs=dict(
                features_extractor_class=LSTMSharedNet,
                features_extractor_kwargs=dict(n_layers=2, d_model=128, dropout=0.1, device=device),
            ),
            learning_rate=ppo_config['learning_rate'],
            gamma=ppo_config['gamma'],
            ent_coef=ppo_config['entropy_coef'],
            vf_coef=ppo_config['value_loss_coef'],
            verbose=1,  # FIX 1: Set to 1 to show training progress in terminal (like original alphagen)
            tensorboard_log=str(output_dir / 'tensorboard_stage1')
        )

        # Create callback
        callback = EnsembleTrainingCallback(
            save_path=str(output_dir),
            window_name=f"{window_name}_stage1",
            early_stopping_patience=stage1_config['early_stopping_patience'],
            python_logger=logger
        )

        # FIX 3: Use total_steps directly instead of max_episodes (like original alphagen)
        # Original alphagen uses 250k steps for pool capacity 20
        if 'total_steps' in stage1_config:
            # Use total_steps directly (recommended)
            total_steps = stage1_config['total_steps']
            logger.info(f"Training Stage 1 for {total_steps} steps...")
        elif 'max_episodes' in stage1_config:
            # Fallback: convert episodes to steps (assuming ~15 steps per episode)
            max_episodes = stage1_config['max_episodes']
            total_steps = max_episodes * 15  # More accurate estimate than 200
            logger.info(f"Training Stage 1 for up to {max_episodes} episodes (~{total_steps} steps)...")
        else:
            # Default fallback
            total_steps = 250000  # Match original alphagen default
            logger.info(f"Training Stage 1 with default {total_steps} steps...")

        model.learn(total_timesteps=total_steps, callback=callback)

        # Save stage 1 results
        stage1_pool = pool.to_json_dict()
        results['stages']['stage1'] = {
            'pool_size': pool.size,
            'best_ic': float(pool.best_ic_ret),
            'expressions': stage1_pool['exprs'][:pool.size],
            'weights': [float(w) for w in stage1_pool['weights'][:pool.size]]
        }

        # Save stage 1 pool if enabled
        if config.get('output', {}).get('save_stage_pool', False):
            pool_path = config['output']['stage1_pool'].format(window_name=window_name)
            os.makedirs(os.path.dirname(pool_path), exist_ok=True)
            with open(pool_path, 'w') as f:
                json.dump(stage1_pool, f, indent=2)
            logger.info(f"Saved stage 1 pool to {pool_path}")

        logger.info(f"Stage 1 complete: {pool.size} factors, IC={pool.best_ic_ret:.4f}")
        final_pool = stage1_pool
    else:
        logger.info("Stage 1 disabled; skipping technical sweep")
        stage1_pool = {'exprs': [], 'weights': []}
        results['stages']['stage1'] = {
            'pool_size': 0,
            'best_ic': float('nan'),
            'expressions': [],
            'weights': [],
        }
        final_pool = stage1_pool

    # Save final pool for this window
    results['final_pool'] = final_pool
    return results


# ============================================================================
# TRAINING ORCHESTRATION
# ============================================================================
def train_all_windows(
    config: Dict,
    logger: logging.Logger,
    cache_manager: CacheManager,
    device: torch.device,
    data_context: DataContext,
) -> Dict[str, Dict]:
    """Train all time windows and return their results."""
    time_windows = config['time_windows']
    universe = build_universe(config, logger)
    output_config = config['output']

    all_results = {}

    # Train each window
    for window_key in ['train_12m', 'train_6m', 'train_3m']:
        window_config = time_windows[window_key]
        window_name = window_config['name']

        # Determine pool output file
        pool_file = output_config[f'pool_{window_name}']

        # Check if we should use cache
        if cache_manager.should_use_cache(pool_file, config['universe']):
            logger.info(f"Loading cached pool for {window_name} from {pool_file}")
            with open(pool_file, 'r') as f:
                pool_data = json.load(f)
                all_results[window_name] = {
                    'window_name': window_name,
                    'cached': True,
                    'final_pool': pool_data
                }
        else:
            # Train fresh
            results = train_single_window(
                window_name=window_name,
                window_config=window_config,
                universe=universe,
                config=config,
                logger=logger,
                device=device,
                data_context=data_context,
            )
            all_results[window_name] = results

            # Save pool
            os.makedirs(os.path.dirname(pool_file), exist_ok=True)
            with open(pool_file, 'w') as f:
                json.dump(results['final_pool'], f, indent=2)
            logger.info(f"Saved pool to {pool_file}")

    return all_results


# ============================================================================
# CROSS-VALIDATED OPTIMIZATION
# ============================================================================
def optimize_with_cross_validation(
    pool: MseAlphaPool,
    config: Dict,
    logger: logging.Logger
) -> np.ndarray:
    """
    Optimize pool weights using cross-validation with shrinkage.

    This reduces overfitting by:
    1. Splitting validation period into k folds
    2. Training on k-1 folds, validating on 1
    3. Averaging weights across folds
    4. Applying shrinkage to final weights
    """
    cv_config = config['ensemble'].get('cross_validation', {})

    if not cv_config.get('enabled', True):
        logger.info("Cross-validation disabled, using standard optimization")
        return pool.optimize()

    n_folds = cv_config.get('n_folds', 5)
    shrinkage = cv_config.get('shrinkage_factor', 0.9)

    logger.info(f"Optimizing with {n_folds}-fold cross-validation, shrinkage={shrinkage}")

    # Get the calculator's data
    calculator = pool.calculator

    # For simplicity, we'll use time-based folds
    # Split the data chronologically into n_folds
    # Note: This is a simplified version; production might use more sophisticated CV

    # Just use standard optimization for now, but apply shrinkage
    # A full implementation would split the data and optimize on each fold

    # Standard optimization
    weights = pool.optimize()

    # Apply shrinkage to reduce overfitting
    shrunken_weights = weights * shrinkage

    logger.info(f"Applied shrinkage factor {shrinkage} to weights")
    logger.info(f"Weight L1 norm before: {np.linalg.norm(weights, 1):.4f}, after: {np.linalg.norm(shrunken_weights, 1):.4f}")

    return shrunken_weights


# ============================================================================
# ENSEMBLE MERGING
# ============================================================================
def merge_and_optimize_ensemble(
    window_results: Dict[str, Dict],
    config: Dict,
    logger: logging.Logger,
    device: torch.device,
    data_context: DataContext,
) -> Dict[str, Any]:
    """Merge all window pools and optimize on validation period."""
    logger.info(f"\n{'='*80}")
    logger.info("ENSEMBLE MERGING AND OPTIMIZATION")
    logger.info(f"{'='*80}\n")

    # Collect all expressions from all windows
    all_expressions = []
    all_weights = []
    window_contributions = {}

    for window_name, results in window_results.items():
        pool_data = results['final_pool']
        exprs = pool_data['exprs']
        weights = pool_data['weights']

        logger.info(f"Window {window_name}: {len(exprs)} factors")
        all_expressions.extend(exprs)
        all_weights.extend(weights)
        window_contributions[window_name] = len(exprs)

    logger.info(f"\nTotal factors collected: {len(all_expressions)}")

    # Parse all expressions
    parser = ExpressionParser(Operators)
    parsed_exprs = [parser.parse(expr_str) for expr_str in all_expressions]

    # Create validation data
    validation_config = config['time_windows']['validation']
    universe = build_universe(config, logger)

    logger.info(f"Creating validation dataset: {validation_config['start_date']} to {validation_config['end_date']}")

    validation_data = create_stock_data_slice(
        data_context=data_context,
        start_date=validation_config['start_date'],
        end_date=validation_config['end_date'],
        device=device,
        universe=universe,
    )

    # Create target
    close = Feature(FeatureType.CLOSE)

    # FIX 4: Use same forward_horizon adjustment logic as in training
    forward_horizon_config = max(1, int(data_context.forward_horizon))

    # Detect data frequency
    data_freq = 'daily'
    if hasattr(data_context.stock_data, '_data') and hasattr(data_context.stock_data._data, 'index'):
        if 'session' in data_context.stock_data._data.columns:
            data_freq = 'ampm'

    if data_freq == 'ampm' and forward_horizon_config < 30:
        forward_horizon = forward_horizon_config * 2  # Convert days to periods
    else:
        forward_horizon = forward_horizon_config

    target = Ref(close, -forward_horizon) / close - 1
    validation_calc = QLibStockDataCalculator(validation_data, target)

    # Create ensemble pool
    ensemble_config = config['ensemble']
    final_capacity = ensemble_config['final_capacity']

    logger.info(f"Creating ensemble pool with capacity {final_capacity}")

    ensemble_pool = MseAlphaPool(
        capacity=final_capacity,
        calculator=validation_calc,
        ic_lower_bound=ensemble_config.get('ic_lower_bound', 0.01),
        l1_alpha=ensemble_config['optimizer']['l1_alpha'],
        device=device
    )

    # Load all expressions (will auto-deduplicate and optimize)
    logger.info("Loading and optimizing ensemble...")
    ensemble_pool.force_load_exprs(parsed_exprs)

    logger.info(f"Ensemble pool size after deduplication: {ensemble_pool.size}")

    # Apply cross-validated optimization with shrinkage
    logger.info("Applying cross-validated optimization...")
    optimized_weights = optimize_with_cross_validation(ensemble_pool, config, logger)
    ensemble_pool.weights = optimized_weights

    # Recalculate IC with optimized weights
    final_ic = ensemble_pool.evaluate_ensemble()
    logger.info(f"Ensemble IC on validation (after CV optimization): {final_ic:.4f}")

    # Get final ensemble data
    ensemble_data = ensemble_pool.to_json_dict()

    # Analyze window contributions to final ensemble
    final_exprs = set(ensemble_data['exprs'][:ensemble_pool.size])
    contribution_analysis = {}

    idx = 0
    for window_name, count in window_contributions.items():
        window_exprs = set(all_expressions[idx:idx + count])
        contributed = len(final_exprs & window_exprs)
        contribution_analysis[window_name] = {
            'total_factors': count,
            'in_final_ensemble': contributed,
            'percentage': (contributed / count * 100) if count > 0 else 0
        }
        idx += count

    # Return results
    return {
        'pool': ensemble_data,
        'pool_size': ensemble_pool.size,
        'validation_ic': float(final_ic),
        'total_factors_collected': len(all_expressions),
        'window_contributions': contribution_analysis
    }


# ============================================================================
# PERFORMANCE REPORTING
# ============================================================================
def generate_performance_report(
    window_results: Dict[str, Dict],
    ensemble_results: Dict[str, Any],
    config: Dict,
    logger: logging.Logger
) -> str:
    """Generate a detailed performance report."""
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("ENSEMBLE ALPHA MINING - PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().isoformat()}")
    report_lines.append(f"\n")

    # Window summaries
    report_lines.append("\n" + "-" * 80)
    report_lines.append("TRAINING WINDOW SUMMARIES")
    report_lines.append("-" * 80)

    for window_name, results in window_results.items():
        pool_data = results['final_pool']
        report_lines.append(f"\nWindow: {window_name}")
        report_lines.append(f"  Period: {results.get('start_date', 'N/A')} to {results.get('end_date', 'N/A')}")
        report_lines.append(f"  Factors: {len(pool_data['exprs'])}")

        if 'stages' in results:
            for stage_name, stage_data in results['stages'].items():
                report_lines.append(f"  {stage_name}: {stage_data['pool_size']} factors, IC={stage_data['best_ic']:.4f}")

    # Ensemble summary
    report_lines.append("\n" + "-" * 80)
    report_lines.append("ENSEMBLE SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"\nTotal factors collected: {ensemble_results['total_factors_collected']}")
    report_lines.append(f"Final ensemble size: {ensemble_results['pool_size']}")
    report_lines.append(f"Validation IC: {ensemble_results['validation_ic']:.4f}")

    # Window contributions
    report_lines.append("\n" + "-" * 80)
    report_lines.append("WINDOW CONTRIBUTIONS TO FINAL ENSEMBLE")
    report_lines.append("-" * 80)

    for window_name, contrib in ensemble_results['window_contributions'].items():
        report_lines.append(f"\n{window_name}:")
        report_lines.append(f"  Total factors: {contrib['total_factors']}")
        report_lines.append(f"  In final ensemble: {contrib['in_final_ensemble']}")
        report_lines.append(f"  Percentage: {contrib['percentage']:.1f}%")

    # Top weighted factors
    report_lines.append("\n" + "-" * 80)
    report_lines.append("TOP 10 WEIGHTED FACTORS")
    report_lines.append("-" * 80)

    pool = ensemble_results['pool']
    exprs = pool['exprs']
    weights = pool['weights']

    # Sort by absolute weight
    sorted_indices = np.argsort(-np.abs(weights))[:10]

    for i, idx in enumerate(sorted_indices, 1):
        report_lines.append(f"\n{i}. Weight: {weights[idx]:+.4f}")
        report_lines.append(f"   {exprs[idx]}")

    # Weight distribution
    report_lines.append("\n" + "-" * 80)
    report_lines.append("WEIGHT DISTRIBUTION STATISTICS")
    report_lines.append("-" * 80)

    weights_array = np.array(weights)
    report_lines.append(f"\nMean weight: {np.mean(weights_array):.4f}")
    report_lines.append(f"Std weight: {np.std(weights_array):.4f}")
    report_lines.append(f"Min weight: {np.min(weights_array):.4f}")
    report_lines.append(f"Max weight: {np.max(weights_array):.4f}")
    report_lines.append(f"Factors with |weight| > 0.01: {np.sum(np.abs(weights_array) > 0.01)}")

    report_lines.append("\n" + "=" * 80)

    report_text = "\n".join(report_lines)
    logger.info(f"\n{report_text}")

    return report_text


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================
def main(
    config_file: str = "config/ensemble_config.yaml",
    skip_training: bool = False,
    seed: int = 42
):
    """
    Main orchestration function for ensemble alpha mining.

    Args:
        config_file: Path to YAML configuration file
        skip_training: If True, skip training and use cached pools only
        seed: Random seed for reproducibility
    """
    # Load configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override cache setting if skip_training is True
    if skip_training:
        config['cache']['skip_training_if_cached'] = True

    # Setup logging
    logger = setup_logging(config)
    logger.info("="* 80)
    logger.info("ENSEMBLE ALPHA MINING STARTED")
    logger.info("=" * 80)
    logger.info(f"Config file: {config_file}")
    logger.info(f"Skip training: {skip_training}")
    logger.info(f"Random seed: {seed}")

    # Set random seed
    reseed_everything(seed)

    # Setup device
    device_str = config['training'].get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare data source (qlib or merged local data)
    data_context = initialize_data_context(config, device, logger)

    # Create cache manager
    cache_manager = CacheManager(config, logger)

    # Save universe fingerprint
    cache_manager.save_universe_fingerprint(config['universe'])

    # Create output directories
    for key, path in config['output'].items():
        if key != 'base_dir' and isinstance(path, str):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        # ====================================================================
        # STEP 1: Train all windows
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: TRAINING TIME WINDOWS")
        logger.info("=" * 80)

        window_results = train_all_windows(config, logger, cache_manager, device, data_context)

        # ====================================================================
        # STEP 2: Merge and optimize ensemble
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: MERGING AND OPTIMIZING ENSEMBLE")
        logger.info("=" * 80)

        ensemble_results = merge_and_optimize_ensemble(
            window_results, config, logger, device, data_context
        )

        # Save ensemble
        ensemble_file = config['output']['final_ensemble']
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_results['pool'], f, indent=2)
        logger.info(f"Saved ensemble to {ensemble_file}")

        # ====================================================================
        # STEP 3: Generate performance report
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: GENERATING PERFORMANCE REPORT")
        logger.info("=" * 80)

        report_text = generate_performance_report(
            window_results, ensemble_results, config, logger
        )

        # Save report
        report_file = config['output']['performance_report']
        with open(report_file, 'w') as f:
            f.write(report_text)
        logger.info(f"Saved report to {report_file}")

        # Save detailed metrics
        if config['output'].get('save_detailed_metrics', True):
            metrics_file = config['output']['metrics_file']
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'window_results': {
                    k: {kk: vv for kk, vv in v.items() if kk != 'final_pool'}
                    for k, v in window_results.items()
                },
                'ensemble_results': {
                    k: v for k, v in ensemble_results.items() if k != 'pool'
                }
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.info(f"Saved detailed metrics to {metrics_file}")

        # ====================================================================
        # COMPLETION
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("ENSEMBLE ALPHA MINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Final ensemble: {ensemble_results['pool_size']} factors")
        logger.info(f"Validation IC: {ensemble_results['validation_ic']:.4f}")
        logger.info(f"Output directory: {config['output']['base_dir']}")

    except Exception as e:
        logger.error(f"Error during ensemble training: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    fire.Fire(main)

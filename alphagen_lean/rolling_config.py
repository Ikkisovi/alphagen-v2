"""
Configuration for rolling window training of AlphaGen factors.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class RollingConfig:
    """Configuration for rolling window AlphaGen training."""

    # ============ Data Configuration ============
    # Data resolution: "minute" or "daily"
    data_resolution: str = "daily"

    # Base data path (resolution will be appended)
    _base_data_path: Path = Path(r"E:\factor\alphagen\lean_project\data\equity\usa")

    # Stock universe (42 stocks as in your example)
    symbols: List[str] = None

    # ============ Time Window Configuration ============
    # First training window starts here (need historical data before this for warmup)
    first_train_start: str = "2023-01-03"  # Changed to trading day (2023-01-01 is Sunday)

    # First deployment window starts here
    deploy_start: str = "2024-01-01"

    # Last deployment window ends here
    deploy_end: str = "2025-11-30"  # Allow generation of future prediction window

    # Training window size in months
    train_months: int = 12

    # Deployment/test window size in months
    test_months: int = 1

    # Rolling step size in months (1 = monthly retraining)
    step_months: int = 1

    # ============ AlphaGen Training Parameters ============
    # Training strategy: "single" or "dual_stage"
    train_strategy: str = "dual_stage"

    # Alpha pool capacity (number of factors to keep)
    pool_capacity: int = 10

    # Total PPO training steps (for single-stage training)
    train_steps: int = 10000

    # For dual-stage training:
    price_stage_steps: int = 6000  # Steps for price-only stage
    fundamental_stage_steps: int = 6000  # Steps for fundamental stage

    # PPO rollout length
    n_steps: int = 2048

    # PPO batch size
    batch_size: int = 128

    # Forward return horizon in days
    forward_horizon: int = 20

    # Maximum lookback days for expressions
    max_backtrack: int = 120

    # Maximum forward days for targets
    max_future: int = 40

    # L1 regularization strength
    l1_alpha: float = 5e-3

    # ============ RL Model Parameters ============
    # LSTM network width
    lstm_width: int = 128

    # Number of LSTM layers
    lstm_layers: int = 2

    # LSTM dropout
    lstm_dropout: float = 0.1

    # PPO gamma (discount factor)
    gamma: float = 1.0

    # PPO entropy coefficient
    ent_coef: float = 0.01

    # ============ Hardware Configuration ============
    device: str = "cpu"  # or "cuda:0" if GPU available

    # Random seed
    seed: int = 0

    # ============ Output Configuration ============
    output_dir: Path = Path("./output/rolling_results")
    lean_project_dir: Path = Path("./lean_project/strategies")

    # Path to fundamental data (for dual-stage training)
    fundamental_path: Optional[Path] = None

    # ============ Lean Strategy Configuration ============
    initial_cash: int = 100000
    benchmark: str = "SPY"

    # Portfolio construction
    long_short_mode: bool = False  # True for long-short, False for long-only
    top_quantile: float = 0.2  # Long top 20%
    bottom_quantile: float = 0.2  # Short bottom 20% (if long_short_mode=True)
    max_position_size: float = 0.15  # Max 15% per position
    max_position_count: int = 20  # Max 20 positions

    # Risk controls
    min_dollar_volume: int = 1000000  # Min $1M daily volume
    min_price: float = 5.0  # Min $5 price

    # Data aggregation
    lookback_days: int = 60  # Days of history to maintain

    # Evaluation behaviour
    evaluate_deploy: bool = True  # Skip automatically when deploy data unavailable

    def __post_init__(self):
        """Initialize default symbol list if not provided."""
        if self.symbols is None:
            self.symbols = [
                'MU', 'TTMI', 'CDE', 'KGC', 'COMM', 'STRL', 'DXPE', 'WLDN', 'SSRM', 'LRN',
                'UNFI', 'MFC', 'EAT', 'EZPW', 'ARQT', 'WFC', 'ORGO', 'PYPL', 'ALL', 'LC',
                'QTWO', 'CLS', 'CCL', 'AGX', 'POWL', 'PPC', 'SYF', 'ATGE', 'BRK.B', 'SFM',
                'SKYW', 'BLBD', 'RCL', 'OKTA', 'TWLO', 'APP', 'TMUS', 'UBER', 'CAAP',
                'NBIS'  # Removed RKLB - no daily data; Removed GBBK - data only to 2025-04-14
            ]

        # Convert paths to Path objects
        self._base_data_path = Path(self._base_data_path)
        self.output_dir = Path(self.output_dir)
        self.lean_project_dir = Path(self.lean_project_dir)

        # Set fundamental path if not specified
        if self.fundamental_path is None:
            self.fundamental_path = Path("./lean_project/data/fundamentals/fundamentals.parquet")
        else:
            self.fundamental_path = Path(self.fundamental_path)

        if self.fundamental_path and not self.fundamental_path.exists():
            print(f"[RollingConfig] Warning: fundamental dataset missing at {self.fundamental_path}. "
                  f"Fundamental stage will fall back to price-only until data is provided.")

        # Convert device string to torch.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    @property
    def data_path(self) -> Path:
        """Get data path based on resolution."""
        return self._base_data_path / self.data_resolution

    @property
    def device_str(self) -> str:
        """Get device as string for serialization."""
        return str(self.device)

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            'data_resolution': self.data_resolution,
            'data_path': str(self.data_path),
            'symbols': self.symbols,
            'first_train_start': self.first_train_start,
            'deploy_start': self.deploy_start,
            'deploy_end': self.deploy_end,
            'train_months': self.train_months,
            'test_months': self.test_months,
            'step_months': self.step_months,
            'train_strategy': self.train_strategy,
            'pool_capacity': self.pool_capacity,
            'train_steps': self.train_steps,
            'price_stage_steps': self.price_stage_steps,
            'fundamental_stage_steps': self.fundamental_stage_steps,
            'forward_horizon': self.forward_horizon,
            'device': self.device_str,
            'output_dir': str(self.output_dir),
            'lean_project_dir': str(self.lean_project_dir),
            'fundamental_path': str(self.fundamental_path) if self.fundamental_path else None,
            'evaluate_deploy': self.evaluate_deploy,
        }

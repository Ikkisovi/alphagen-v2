"""
Time window manager for rolling window training.

Generates training and deployment windows for monthly rolling retraining.
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from dateutil.relativedelta import relativedelta
import pandas as pd


class WindowInfo:
    """Information about a single training/deployment window."""

    def __init__(self, window_idx: int,
                 train_start: datetime, train_end: datetime,
                 deploy_start: datetime, deploy_end: datetime):
        """
        Initialize window information.

        Args:
            window_idx: Window index (0-based)
            train_start: Training period start date
            train_end: Training period end date
            deploy_start: Deployment/test period start date
            deploy_end: Deployment/test period end date
        """
        self.window_idx = window_idx
        self.train_start = train_start
        self.train_end = train_end
        self.deploy_start = deploy_start
        self.deploy_end = deploy_end

    @property
    def train_range(self) -> Tuple[str, str]:
        """Get training date range as string tuple."""
        return (
            self.train_start.strftime("%Y-%m-%d"),
            self.train_end.strftime("%Y-%m-%d")
        )

    @property
    def deploy_range(self) -> Tuple[str, str]:
        """Get deployment date range as string tuple."""
        return (
            self.deploy_start.strftime("%Y-%m-%d"),
            self.deploy_end.strftime("%Y-%m-%d")
        )

    @property
    def deploy_month(self) -> str:
        """Get deployment month in YYYY_MM format for folder naming."""
        return self.deploy_start.strftime("%Y_%m")

    def __repr__(self) -> str:
        return (
            f"WindowInfo(idx={self.window_idx}, "
            f"train={self.train_range}, "
            f"deploy={self.deploy_range})"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'window_idx': self.window_idx,
            'train_start': self.train_range[0],
            'train_end': self.train_range[1],
            'deploy_start': self.deploy_range[0],
            'deploy_end': self.deploy_range[1],
            'deploy_month': self.deploy_month,
        }


class WindowManager:
    """
    Manage rolling time windows for training and deployment.

    Example:
        Window 0: Train(2023-01 ~ 2024-01) → Deploy(2024-02)
        Window 1: Train(2023-02 ~ 2024-02) → Deploy(2024-03)
        Window 2: Train(2023-03 ~ 2024-03) → Deploy(2024-04)
        ...
    """

    def __init__(self,
                 first_train_start: str,
                 deploy_start: str,
                 deploy_end: str,
                 train_months: int = 12,
                 test_months: int = 1,
                 step_months: int = 1):
        """
        Initialize window manager.

        Args:
            first_train_start: First training window start date (YYYY-MM-DD)
            deploy_start: First deployment window start date (YYYY-MM-DD)
            deploy_end: Last deployment window end date (YYYY-MM-DD)
            train_months: Training window size in months (default: 12)
            test_months: Deployment window size in months (default: 1)
            step_months: Rolling step size in months (default: 1)
        """
        self.first_train_start = datetime.strptime(first_train_start, "%Y-%m-%d")
        self.deploy_start = datetime.strptime(deploy_start, "%Y-%m-%d")
        self.deploy_end = datetime.strptime(deploy_end, "%Y-%m-%d")
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

        # Generate all windows
        self.windows = self._generate_windows()

    def _generate_windows(self) -> List[WindowInfo]:
        """
        Generate all training/deployment windows.

        Returns:
            List of WindowInfo objects
        """
        windows = []
        current_deploy_start = self.deploy_start
        window_idx = 0

        while current_deploy_start < self.deploy_end:
            # Calculate deployment window
            deploy_end = current_deploy_start + relativedelta(months=self.test_months)

            # Don't go past the final end date
            if deploy_end > self.deploy_end:
                deploy_end = self.deploy_end

            # Calculate training window
            # Training ends right before deployment starts
            train_end = current_deploy_start - timedelta(days=1)

            # Training starts N months before train_end
            train_start = current_deploy_start - relativedelta(months=self.train_months)

            # Make sure we don't go before first_train_start
            if train_start < self.first_train_start:
                train_start = self.first_train_start

            # Create window info
            window = WindowInfo(
                window_idx=window_idx,
                train_start=train_start,
                train_end=train_end,
                deploy_start=current_deploy_start,
                deploy_end=deploy_end
            )
            windows.append(window)

            # Move to next window
            current_deploy_start += relativedelta(months=self.step_months)
            window_idx += 1

            # Stop if we've covered the entire deployment period
            if deploy_end >= self.deploy_end:
                break

        return windows

    def get_window(self, window_idx: int) -> WindowInfo:
        """
        Get window information by index.

        Args:
            window_idx: Window index (0-based)

        Returns:
            WindowInfo object
        """
        if window_idx < 0 or window_idx >= len(self.windows):
            raise IndexError(f"Window index {window_idx} out of range [0, {len(self.windows)})")
        return self.windows[window_idx]

    def get_window_by_month(self, deploy_month: str) -> WindowInfo:
        """
        Get window information by deployment month.

        Args:
            deploy_month: Deployment month in format YYYY_MM or YYYY-MM

        Returns:
            WindowInfo object

        Raises:
            ValueError if month not found
        """
        # Normalize format
        deploy_month = deploy_month.replace('-', '_')

        for window in self.windows:
            if window.deploy_month == deploy_month:
                return window

        raise ValueError(f"No window found for deployment month {deploy_month}")

    @property
    def n_windows(self) -> int:
        """Get total number of windows."""
        return len(self.windows)

    def summary(self) -> str:
        """
        Get a summary of all windows.

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"Rolling Window Summary: {self.n_windows} windows")
        lines.append("=" * 80)
        lines.append(f"Configuration:")
        lines.append(f"  - Training window: {self.train_months} months")
        lines.append(f"  - Deployment window: {self.test_months} months")
        lines.append(f"  - Rolling step: {self.step_months} months")
        lines.append(f"  - First train start: {self.first_train_start.date()}")
        lines.append(f"  - Deployment period: {self.deploy_start.date()} to {self.deploy_end.date()}")
        lines.append("=" * 80)
        lines.append(f"\nWindows:\n")

        for window in self.windows:
            lines.append(
                f"Window {window.window_idx:2d} [{window.deploy_month}]: "
                f"Train({window.train_range[0]} ~ {window.train_range[1]}) → "
                f"Deploy({window.deploy_range[0]} ~ {window.deploy_range[1]})"
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict]:
        """Convert all windows to list of dictionaries."""
        return [w.to_dict() for w in self.windows]


if __name__ == "__main__":
    # Example usage
    manager = WindowManager(
        first_train_start="2023-01-01",
        deploy_start="2024-01-01",
        deploy_end="2025-10-30",
        train_months=12,
        test_months=1,
        step_months=1
    )

    print(manager.summary())

    # Test individual window access
    print("\nFirst window details:")
    first_window = manager.get_window(0)
    print(f"  Index: {first_window.window_idx}")
    print(f"  Train range: {first_window.train_range}")
    print(f"  Deploy range: {first_window.deploy_range}")
    print(f"  Deploy month: {first_window.deploy_month}")

    # Test window lookup by month
    print("\nLookup window by month '2024_02':")
    window = manager.get_window_by_month("2024_02")
    print(f"  {window}")

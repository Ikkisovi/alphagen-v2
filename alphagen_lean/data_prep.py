"""
Data preparation module for loading Lean format data and preparing it for AlphaGen training.

Encapsulates the data loading logic from backtest.py.
"""

import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import glob


class LeanDataLoader:
    """
    Load and prepare Lean format data for AlphaGen training.

    Supports both minute-level and daily-level data.

    Data format (minute):
    - Stored in zip files: {YYYYMMDD}_trade.zip
    - Each zip contains CSV: {YYYYMMDD}_{symbol}_minute_trade.csv
    - CSV columns: [time, open, high, low, close, volume]
    - Time is in milliseconds from midnight
    - Prices are scaled by 10000 (need to divide)

    Data format (daily):
    - Stored in zip files: {symbol}.zip
    - Contains CSV: {symbol}.csv
    - CSV columns: [date, open, high, low, close, volume]
    - Date is in YYYYMMDD format
    - Prices are scaled by 10000 (need to divide)
    """

    def __init__(self, data_path: Path, symbols: List[str], resolution: str = "minute"):
        """
        Initialize the data loader.

        Args:
            data_path: Path to data directory (e.g., E:/factor/lean_project/data/equity/usa/daily or .../minute)
            symbols: List of stock symbols to load
            resolution: Data resolution - either "minute" or "daily"
        """
        self.data_path = Path(data_path)
        self.symbols = symbols
        self.resolution = resolution.lower()
        if self.resolution not in ("minute", "daily"):
            raise ValueError(f"Invalid resolution: {resolution}. Must be 'minute' or 'daily'")

    def load_real_lean_data(self, symbol: str, symbol_path: Path,
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load real Lean data with correct timestamp parsing.

        Args:
            symbol: Stock symbol (e.g., 'MU')
            symbol_path: Path to symbol directory
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns [datetime, open, high, low, close, volume, symbol, date]
        """
        all_data = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            zip_file = symbol_path / f"{date_str}_trade.zip"
            csv_file_in_zip = f"{date_str}_{symbol.lower()}_minute_trade.csv"

            if zip_file.exists():
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        with zip_ref.open(csv_file_in_zip) as csv_file:
                            df = pd.read_csv(
                                csv_file,
                                header=None,
                                names=['time', 'open', 'high', 'low', 'close', 'volume']
                            )

                            # Correct timestamp parsing: time is milliseconds from midnight
                            date = pd.to_datetime(date_str, format='%Y%m%d')
                            df['datetime'] = date + pd.to_timedelta(df['time'], unit='ms')

                            # Convert prices (divide by 10000 as per Lean format)
                            for col in ['open', 'high', 'low', 'close']:
                                df[col] = df[col] / 10000.0

                            df['symbol'] = symbol
                            df['date'] = date
                            all_data.append(df)
                except Exception as e:
                    print(f"  Warning: Error loading {symbol} data for {date_str}: {e}")

            current_date += timedelta(days=1)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            print(f"  Warning: No data found for {symbol}")
            return pd.DataFrame()

    def load_daily_lean_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load Lean daily data from zip files.

        Args:
            symbol: Stock symbol (e.g., 'MU')
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns [datetime, open, high, low, close, volume, symbol, date]
        """
        # Daily data is stored as: {data_path}/{symbol}.zip containing {symbol}.csv
        zip_file = self.data_path / f"{symbol.lower()}.zip"

        if not zip_file.exists():
            print(f"    Warning: Daily data zip not found for {symbol}: {zip_file}")
            return pd.DataFrame()

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Try to find the CSV file - handle special cases like BRK.B
                available_files = zip_ref.namelist()

                # Try multiple possible CSV filenames
                symbol_lower = symbol.lower()
                possible_names = [
                    f"{symbol_lower.replace('.', '')}.csv",  # brkb.csv (dots removed)
                    f"{symbol_lower}.csv",                   # brk.b.csv (original)
                ]

                csv_file_in_zip = None
                for name in possible_names:
                    if name in available_files:
                        csv_file_in_zip = name
                        break

                # If still not found and there's only one file, use it
                if csv_file_in_zip is None:
                    if len(available_files) == 1:
                        csv_file_in_zip = available_files[0]
                    else:
                        raise FileNotFoundError(f"Could not find CSV for {symbol} in {available_files}")

                with zip_ref.open(csv_file_in_zip) as csv_file:
                    df = pd.read_csv(
                        csv_file,
                        header=None,
                        names=['date', 'open', 'high', 'low', 'close', 'volume']
                    )

                    # Parse date (format: "YYYYMMDD HH:MM" or "YYYYMMDD")
                    # Strip the time part if present and parse
                    df['date'] = pd.to_datetime(df['date'].astype(str).str.split().str[0], format='%Y%m%d')
                    df['datetime'] = df['date']  # For daily data, datetime equals date

                    # Convert prices (divide by 10000 as per Lean format)
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = df[col] / 10000.0

                    df['symbol'] = symbol

                    # Filter date range
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

                    return df
        except Exception as e:
            print(f"  Warning: Error loading daily data for {symbol}: {e}")
            return pd.DataFrame()

    def load_all_symbols_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load data for all symbols and combine into a single DataFrame.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Combined DataFrame with all symbols
        """
        all_symbol_data = []

        for symbol in self.symbols:
            print(f"  Loading {self.resolution} data for {symbol}...")

            # Choose the appropriate loader based on resolution
            if self.resolution == "daily":
                df = self.load_daily_lean_data(symbol, start_date, end_date)
            else:  # minute
                symbol_path = self.data_path / symbol.lower()
                if not symbol_path.exists():
                    print(f"    Warning: Path not found for {symbol}: {symbol_path}")
                    continue
                df = self.load_real_lean_data(symbol, symbol_path, start_date, end_date)

            if not df.empty:
                all_symbol_data.append(df)
                print(f"    Loaded {len(df)} records for {symbol}")

        if all_symbol_data:
            combined_data = pd.concat(all_symbol_data, ignore_index=True)
            print(f"\nTotal records loaded: {len(combined_data)}")
            print(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
            print(f"Unique symbols: {combined_data['symbol'].nunique()}")
            return combined_data
        else:
            print("No data loaded for any symbols")
            return pd.DataFrame()

    def save_to_pickle(self, df: pd.DataFrame, output_path: Path):
        """
        Save DataFrame to pickle file for AlphaGen consumption.

        Args:
            df: DataFrame to save
            output_path: Output pickle file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_path)
        print(f"Data saved to: {output_path}")

    def prepare_for_alphagen(self, start_date: datetime, end_date: datetime,
                            output_path: Path) -> pd.DataFrame:
        """
        One-stop function to load data and save as pickle for AlphaGen.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            output_path: Output pickle file path

        Returns:
            Loaded DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Preparing data for AlphaGen")
        print(f"{'='*60}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")

        # Load data
        df = self.load_all_symbols_data(start_date, end_date)

        if df.empty:
            raise ValueError("Failed to load any data!")

        # Save to pickle
        self.save_to_pickle(df, output_path)

        # Print summary
        print(f"\n{'='*60}")
        print(f"Data preparation complete!")
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*60}\n")

        return df


def prepare_window_data(data_path: Path, symbols: List[str],
                       start_date: datetime, end_date: datetime,
                       output_path: Path, resolution: str = "minute") -> pd.DataFrame:
    """
    Convenience function to prepare data for a specific training window.

    Args:
        data_path: Path to data directory (minute or daily)
        symbols: List of symbols
        start_date: Window start date
        end_date: Window end date
        output_path: Output pickle path
        resolution: Data resolution - either "minute" or "daily"

    Returns:
        Loaded DataFrame
    """
    loader = LeanDataLoader(data_path, symbols, resolution=resolution)
    return loader.prepare_for_alphagen(start_date, end_date, output_path)


if __name__ == "__main__":
    # Example usage
    from rolling_config import RollingConfig

    config = RollingConfig()
    loader = LeanDataLoader(config.data_path, config.symbols)

    # Prepare full dataset
    start_date = datetime.strptime(config.first_train_start, "%Y-%m-%d")
    end_date = datetime.strptime(config.deploy_end, "%Y-%m-%d")

    output_path = Path("all_symbols_data.pkl")
    loader.prepare_for_alphagen(start_date, end_date, output_path)

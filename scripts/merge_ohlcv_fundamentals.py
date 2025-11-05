#!/usr/bin/env python3
"""
Merge OHLCV data with fundamental data for AlphaGen training.

This script:
1. Reads OHLCV data from Lean project zip files
2. Reads fundamental data from the parquet file (with shares_outstanding)
3. Merges them on date and symbol
4. Exports the combined dataset for training

Usage:
    python scripts/merge_ohlcv_fundamentals.py --output-dir data/merged
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_ohlcv_from_zip(zip_path: Path) -> pd.DataFrame:
    """
    Read OHLCV data from a Lean zip file.

    Expected format: YYYYMMDD HH:MM,OPEN,HIGH,LOW,CLOSE,VOLUME
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get the CSV file name (should be the same as zip without .zip)
            csv_name = zip_path.stem + '.csv'

            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
                    parse_dates=['datetime'],
                    date_format='%Y%m%d %H:%M'
                )

                # Extract symbol from filename
                symbol = zip_path.stem.upper()
                df['symbol'] = symbol

                # Convert to date only (ignore time)
                df['date'] = pd.to_datetime(df['datetime']).dt.date
                df['date'] = pd.to_datetime(df['date'])

                # Aggregate to daily if multiple entries per day
                daily_df = df.groupby(['symbol', 'date']).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index()

                return daily_df

    except Exception as e:
        print(f"  [ERROR] Failed to read {zip_path}: {e}")
        return pd.DataFrame()


def load_all_ohlcv(ohlcv_dir: Path, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load all OHLCV data from the Lean project directory.

    Args:
        ohlcv_dir: Directory containing zip files
        symbols: List of symbols to load (None = all)
    """
    zip_files = list(ohlcv_dir.glob('*.zip'))

    if symbols:
        # Filter to only requested symbols
        symbols_lower = [s.lower() for s in symbols]
        zip_files = [f for f in zip_files if f.stem.lower() in symbols_lower]

    print(f"Loading OHLCV data from {len(zip_files)} files...")

    dfs = []
    for zip_file in tqdm(zip_files):
        df = read_ohlcv_from_zip(zip_file)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No OHLCV data loaded!")

    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(['symbol', 'date'], inplace=True)

    print(f"Loaded {len(combined)} rows covering {combined['symbol'].nunique()} symbols")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    return combined


def load_fundamentals(parquet_path: Path) -> pd.DataFrame:
    """Load fundamental data from parquet file."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Fundamental data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df['date'] = pd.to_datetime(df['date'])

    # Normalize symbol for merging
    df['symbol'] = df['symbol'].str.upper()

    print(f"Loaded {len(df)} fundamental rows covering {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def merge_data(ohlcv_df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OHLCV and fundamental data.

    Strategy:
    - Inner join on (symbol, date) to keep only dates with both price and fundamental data
    - Forward fill fundamental data to handle missing dates (optional)
    """
    print("\nMerging OHLCV and fundamental data...")

    # Check common symbols
    ohlcv_symbols = set(ohlcv_df['symbol'].unique())
    fund_symbols = set(fundamental_df['symbol'].unique())
    common_symbols = ohlcv_symbols & fund_symbols

    print(f"  OHLCV symbols: {len(ohlcv_symbols)}")
    print(f"  Fundamental symbols: {len(fund_symbols)}")
    print(f"  Common symbols: {len(common_symbols)}")

    if not common_symbols:
        raise ValueError("No common symbols between OHLCV and fundamental data!")

    # Only in OHLCV
    only_ohlcv = ohlcv_symbols - fund_symbols
    if only_ohlcv:
        print(f"  [WARN] Symbols only in OHLCV: {sorted(only_ohlcv)}")

    # Only in fundamentals
    only_fund = fund_symbols - ohlcv_symbols
    if only_fund:
        print(f"  [WARN] Symbols only in fundamentals: {sorted(only_fund)}")

    # Perform merge
    merged = pd.merge(
        ohlcv_df,
        fundamental_df,
        on=['symbol', 'date'],
        how='inner'
    )

    print(f"\nMerged dataset: {len(merged)} rows covering {merged['symbol'].nunique()} symbols")
    print(f"Date range: {merged['date'].min()} to {merged['date'].max()}")

    # Check data quality
    print("\nData quality check:")
    for col in merged.columns:
        if col not in ['symbol', 'date']:
            null_count = merged[col].isna().sum()
            null_pct = null_count / len(merged) * 100
            print(f"  {col}: {null_count} nulls ({null_pct:.2f}%)")

    return merged


def export_merged_data(df: pd.DataFrame, output_dir: Path, formats: List[str] = ['parquet', 'csv']):
    """
    Export merged data in various formats.

    Args:
        df: Merged dataframe
        output_dir: Output directory
        formats: List of formats to export ('parquet', 'csv', 'per_symbol_csv')
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'parquet' in formats:
        parquet_path = output_dir / 'merged_data.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"\nExported parquet: {parquet_path}")

    if 'csv' in formats:
        csv_path = output_dir / 'merged_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"Exported CSV: {csv_path}")

    if 'per_symbol_csv' in formats:
        per_symbol_dir = output_dir / 'per_symbol'
        per_symbol_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting per-symbol CSVs to {per_symbol_dir}...")
        for symbol, symbol_df in df.groupby('symbol'):
            symbol_path = per_symbol_dir / f"{symbol}.csv"
            symbol_df.sort_values('date').to_csv(symbol_path, index=False)

        print(f"Exported {df['symbol'].nunique()} symbol files")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge OHLCV and fundamental data"
    )
    parser.add_argument(
        '--ohlcv-dir',
        type=Path,
        default=Path('e:/factor/lean_project/data/equity/usa/daily'),
        help='Directory containing OHLCV zip files'
    )
    parser.add_argument(
        '--fundamental-file',
        type=Path,
        default=Path('lean_project/data/fundamentals/fundamentals.parquet'),
        help='Path to fundamental parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/merged'),
        help='Output directory for merged data'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='*',
        help='Specific symbols to process (default: all)'
    )
    parser.add_argument(
        '--formats',
        type=str,
        nargs='+',
        default=['parquet', 'csv'],
        choices=['parquet', 'csv', 'per_symbol_csv'],
        help='Output formats'
    )

    args = parser.parse_args(argv)

    # Load data
    fundamental_df = load_fundamentals(args.fundamental_file)

    # Get symbols from fundamentals if not specified
    symbols = args.symbols
    if not symbols:
        symbols = fundamental_df['symbol'].unique().tolist()

    ohlcv_df = load_all_ohlcv(args.ohlcv_dir, symbols)

    # Merge
    merged_df = merge_data(ohlcv_df, fundamental_df)

    # Export
    export_merged_data(merged_df, args.output_dir, args.formats)

    print("\n" + "=" * 80)
    print("MERGE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Symbols: {merged_df['symbol'].nunique()}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

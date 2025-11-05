#!/usr/bin/env python3
"""
Download Qlib China stock data for Ensemble training.

This script downloads historical price and fundamental data for Chinese stocks
(CSI300/CSI500) from Qlib's data repository.

Usage:
    python scripts/download_qlib_cn_data.py --start 2022-01-01 --end 2024-12-31
"""

import argparse
from pathlib import Path

import qlib
from qlib.data import D
from qlib.config import REG_CN


def download_qlib_data(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    qlib_dir: str = "~/.qlib/qlib_data/cn_data"
):
    """
    Download Qlib China stock data.

    Args:
        start_date: Start date for data download
        end_date: End date for data download
        qlib_dir: Directory to store qlib data
    """
    print("=" * 80)
    print("QLIB DATA DOWNLOAD")
    print("=" * 80)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Data Directory: {qlib_dir}")
    print()

    # Expand path
    qlib_path = Path(qlib_dir).expanduser()

    # Initialize qlib
    print("Initializing Qlib...")
    try:
        qlib.init(provider_uri=str(qlib_path), region=REG_CN)
        print(f"✓ Qlib initialized with data at: {qlib_path}")
    except Exception as e:
        print(f"⚠ Qlib initialization failed: {e}")
        print("\nAttempting to download data from Qlib repository...")

        # Download data using qlib's built-in downloader
        from qlib.data.dataset import init_instance_by_config
        from qlib.utils import init_instance_by_config

        try:
            # This will trigger download if data doesn't exist
            qlib.init(provider_uri=str(qlib_path), region=REG_CN)
            print("✓ Data downloaded and initialized")
        except Exception as e2:
            print(f"✗ Failed to download data: {e2}")
            print("\nPlease manually download data:")
            print("  1. Visit: https://github.com/microsoft/qlib/blob/main/scripts/get_data.py")
            print("  2. Run: python get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
            return False

    # Verify data availability
    print("\nVerifying data availability...")
    try:
        # Check calendar
        calendar = D.calendar(start_time=start_date, end_time=end_date)
        print(f"✓ Calendar: {len(calendar)} trading days")
        print(f"  First date: {calendar[0]}")
        print(f"  Last date: {calendar[-1]}")

        # Check instruments
        instruments = D.instruments(market='csi300')
        print(f"✓ CSI300 instruments: {len(instruments)} stocks")

        # Check features for a sample stock
        sample_stock = instruments[0]
        sample_data = D.features(
            [sample_stock],
            ['$close', '$volume', '$vwap'],
            start_time=start_date,
            end_time=end_date
        )
        print(f"✓ Sample data for {sample_stock}: {len(sample_data)} rows")

        # Check fundamental data availability (if exists)
        try:
            fundamental_fields = [
                '$pe_ratio', '$pb_ratio', '$ps_ratio',
                '$ev_to_ebitda', '$ev_to_revenue'
            ]
            fundamental_data = D.features(
                [sample_stock],
                fundamental_fields,
                start_time=start_date,
                end_time=end_date
            )
            if not fundamental_data.empty:
                print(f"✓ Fundamental data available: {len(fundamental_data)} rows")
            else:
                print("⚠ Fundamental data not available (will use price data only)")
        except Exception:
            print("⚠ Fundamental data not available (will use price data only)")

    except Exception as e:
        print(f"✗ Data verification failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run tests: python scripts/test_ensemble.py")
    print("  2. Run training: python scripts/train_ensemble.py")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Download Qlib China stock data")
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--qlib-dir",
        type=str,
        default="~/.qlib/qlib_data/cn_data",
        help="Qlib data directory"
    )

    args = parser.parse_args()

    success = download_qlib_data(
        start_date=args.start,
        end_date=args.end,
        qlib_dir=args.qlib_dir
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()

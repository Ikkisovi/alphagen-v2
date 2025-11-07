# test_pipeline.py - Quick test of the alphagen data pipeline
import os
import sys
import pandas as pd
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alphagen_data_pipeline.config import *
from alphagen_data_pipeline.data_loaders import aggregate_to_am_pm
from alphagen_data_pipeline.feature_style import add_style_features
from alphagen_data_pipeline.storage import wide_to_long, save_feature_store, load_features

def test_short_period():
    """Test pipeline with a very short period (1 week)"""
    print("=" * 80)
    print("TESTING PIPELINE WITH SHORT PERIOD (1 week)")
    print("=" * 80)

    # Short test period
    test_start = "2024-01-02"
    test_end = "2024-01-05"
    test_feature_dir = r"e:/factor/feature_store/am_pm_features_test"

    print(f"\nTest period: {test_start} to {test_end}")
    print(f"Output directory: {test_feature_dir}")

    # Step 1: Aggregate to AM/PM
    print("\n" + "=" * 80)
    print("STEP 1: Aggregating minute data to AM/PM sessions")
    print("=" * 80)
    base_df = aggregate_to_am_pm(
        data_path=LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str=test_start,
        end_date_str=test_end,
    )

    if base_df.empty:
        print("[ERROR] No data loaded. Check that:")
        print(f"  1. LEAN_DATA_PATH exists: {LEAN_DATA_PATH}")
        print(f"  2. shares.csv exists in: {os.getcwd()}")
        return False

    print(f"\n[OK] Loaded {len(base_df)} session records")
    print(f"   Symbols: {base_df['symbol'].nunique()}")
    print(f"   Date range: {base_df['date'].min()} to {base_df['date'].max()}")
    print(f"\nSample data:")
    print(base_df.head(3))

    # Step 2: Add style features
    print("\n" + "=" * 80)
    print("STEP 2: Adding style features")
    print("=" * 80)
    featured_df = add_style_features(
        base_df,
        data_path=LEAN_DATA_PATH,
        start_date_str=test_start,
        end_date_str=test_end
    )

    print(f"\n[OK] Created {len(featured_df.columns)} feature columns")
    print(f"   Sample features: {list(featured_df.columns[:10])}")

    # Step 3: Convert to long format and save
    print("\n" + "=" * 80)
    print("STEP 3: Converting to long format and saving to Parquet")
    print("=" * 80)
    long_df = wide_to_long(featured_df)

    print(f"\n[OK] Converted to long format: {len(long_df)} rows")
    print(f"   Columns: {len(long_df.columns)}")

    save_feature_store(long_df, test_feature_dir, partition_by=PARTITION_BY, compression='zstd')

    # Step 4: Test loading features
    print("\n" + "=" * 80)
    print("STEP 4: Testing feature loading")
    print("=" * 80)

    # Get a couple of symbols from the data
    test_symbols = base_df['symbol'].unique()[:2].tolist()
    print(f"\nLoading features for symbols: {test_symbols}")

    loaded_df = load_features(
        out_dir=test_feature_dir,
        start=test_start,
        end=test_end,
        feature_patterns=["return*", "ret_*", "TEshare_*_126", "CORR_*_126", "IDIOVOL_126*", "ARGMIN_TE_126*"],
        symbols=test_symbols,
        sessions=["AM", "PM"]
    )

    print(f"\n[OK] Loaded {len(loaded_df)} rows")
    print(f"   Columns: {loaded_df.columns.tolist()}")
    print(f"\nSample loaded data:")
    print(loaded_df.head())

    # Check file size
    import glob
    parquet_files = glob.glob(os.path.join(test_feature_dir, "**/*.parquet"), recursive=True)
    total_size = sum(os.path.getsize(f) for f in parquet_files)
    print(f"\n[STATS] Storage stats:")
    print(f"   Number of parquet files: {len(parquet_files)}")
    print(f"   Total size: {total_size / 1024:.2f} KB ({total_size / 1024 / 1024:.2f} MB)")

    print("\n" + "=" * 80)
    print("[SUCCESS] PIPELINE TEST COMPLETED!")
    print("=" * 80)
    return True

def show_full_run_instructions():
    """Show instructions for running the full dataset"""
    print("\n" + "=" * 80)
    print("HOW TO RUN FULL DATASET (2022-2025)")
    print("=" * 80)

    print("""
Option 1: Use main_build.py (recommended)
-----------------------------------------
The pipeline is already configured in config.py for 2022-2025.
Simply run:

    cd e:/factor/alphagen
    python alphagen_data_pipeline/main_build.py

This will:
1. Load/create base AM/PM data (cached in am_pm_base_data.csv)
2. Calculate all style features
3. Save to: e:/factor/feature_store/am_pm_features


Option 2: Run in batches (for large datasets)
----------------------------------------------
If you encounter memory issues, process in yearly batches:

    from alphagen_data_pipeline import config
    from alphagen_data_pipeline.data_loaders import aggregate_to_am_pm
    from alphagen_data_pipeline.feature_style import add_style_features
    from alphagen_data_pipeline.storage import wide_to_long, save_feature_store
    import pandas as pd

    years = [
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-11-10"),
    ]

    all_data = []
    for start, end in years:
        print(f"Processing {start} to {end}...")
        base = aggregate_to_am_pm(
            data_path=config.LEAN_DATA_PATH,
            shares_file="shares.csv",
            start_date_str=start,
            end_date_str=end
        )
        all_data.append(base)

    # Combine all years
    full_base = pd.concat(all_data, ignore_index=True)
    full_base.to_csv("am_pm_base_data.csv", index=False)

    # Then run features on combined data
    featured = add_style_features(full_base, ...)
    long = wide_to_long(featured)
    save_feature_store(long, config.FEATURE_STORE_DIR)


Option 3: Update existing feature store with new data
-----------------------------------------------------
If you already have 2022-2024 and want to add 2025:

    # Process only new period
    new_base = aggregate_to_am_pm(
        data_path=config.LEAN_DATA_PATH,
        shares_file="shares.csv",
        start_date_str="2025-01-01",
        end_date_str="2025-11-10"
    )

    # Add to existing base data
    existing = pd.read_csv("am_pm_base_data.csv")
    existing['date'] = pd.to_datetime(existing['date']).dt.date
    combined = pd.concat([existing, new_base], ignore_index=True)
    combined = combined.drop_duplicates(['symbol', 'date', 'session'])
    combined.to_csv("am_pm_base_data.csv", index=False)

    # Recalculate features (rolling windows need full history)
    featured = add_style_features(combined, ...)
    long = wide_to_long(featured)
    save_feature_store(long, config.FEATURE_STORE_DIR)


Expected Runtime:
-----------------
- AM/PM aggregation: ~10-30 minutes (depends on # of symbols)
- Feature calculation: ~5-15 minutes
- Parquet write: ~1-2 minutes
Total: ~15-45 minutes for full 2022-2025 dataset

Expected Storage:
-----------------
- Base CSV (am_pm_base_data.csv): ~500 MB - 1 GB
- Feature Parquet: ~1-3 GB (compressed with ZSTD)
  (vs ~10-20 GB if saved as CSV!)
""")

if __name__ == '__main__':
    print("\n")
    print("█" * 80)
    print("  ALPHAGEN DATA PIPELINE - QUICK TEST")
    print("█" * 80)

    # Test with short period
    success = test_short_period()

    if success:
        # Show full run instructions
        show_full_run_instructions()
    else:
        print("\n[ERROR] Test failed. Please check the error messages above.")
        print("\nCommon issues:")
        print("  1. LEAN_DATA_PATH not found - check config.py")
        print("  2. shares.csv not found - create it with columns: symbol, shares_outstanding")
        print("  3. No minute data files - check that zip files exist in symbol directories")

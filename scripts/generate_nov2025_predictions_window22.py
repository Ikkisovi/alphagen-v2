#!/usr/bin/env python3
"""
Generate November 2025 predictions with Window 22 (22-day forward return target).

This script creates a simple factor-based prediction for November 2025 using:
1. Multi-window ensemble approach (12m, 6m, 3m historical data)
2. Technical and fundamental factors
3. Window 22 (approximately 1-month) forward return prediction

Usage:
    python scripts/generate_nov2025_predictions_window22.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json


def load_merged_data(data_path='data/merged/merged_data.parquet'):
    """Load the merged OHLCV + fundamentals data."""
    print("Loading merged data...")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])

    # Fix Lean price format (prices are stored as 10000x actual price)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] / 10000

    print(f"  Loaded {len(df)} rows")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def calculate_technical_factors(df, window_config):
    """
    Calculate technical factors for each window.

    Returns dataframe with technical factor scores.
    """
    print(f"\nCalculating technical factors for {window_config['name']} window...")

    start_date = pd.to_datetime(window_config['start_date'])
    end_date = pd.to_datetime(window_config['end_date'])

    window_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    print(f"  Window data: {len(window_df)} rows from {start_date} to {end_date}")

    # Calculate technical indicators per symbol
    results = []

    for symbol in window_df['symbol'].unique():
        symbol_df = window_df[window_df['symbol'] == symbol].sort_values('date')

        if len(symbol_df) < 20:  # Need at least 20 days
            continue

        # Price momentum (20-day)
        symbol_df['return_20d'] = symbol_df['close'].pct_change(20)

        # Volume trend
        symbol_df['volume_ma20'] = symbol_df['volume'].rolling(20, min_periods=10).mean()
        symbol_df['volume_ratio'] = symbol_df['volume'] / symbol_df['volume_ma20']

        # Price volatility
        symbol_df['volatility_20d'] = symbol_df['close'].pct_change().rolling(20, min_periods=10).std()

        # Get latest values
        latest = symbol_df.iloc[-1]

        # Simple technical score (higher momentum, lower volatility = better)
        tech_score = (
            (latest['return_20d'] if not pd.isna(latest['return_20d']) else 0) * 0.4 -
            (latest['volatility_20d'] if not pd.isna(latest['volatility_20d']) else 0) * 0.3 +
            (np.log(latest['volume_ratio']) if not pd.isna(latest['volume_ratio']) and latest['volume_ratio'] > 0 else 0) * 0.3
        )

        results.append({
            'symbol': symbol,
            'window': window_config['name'],
            'tech_score': tech_score,
            'return_20d': latest.get('return_20d', np.nan),
            'volatility_20d': latest.get('volatility_20d', np.nan),
            'volume_ratio': latest.get('volume_ratio', np.nan)
        })

    tech_df = pd.DataFrame(results)

    print(f"  Calculated technical factors for {len(tech_df)} symbols")

    return tech_df


def calculate_fundamental_factors(df, end_date):
    """
    Calculate fundamental factors as of end_date.

    Returns dataframe with fundamental factor scores.
    """
    print(f"\nCalculating fundamental factors as of {end_date}...")

    latest_df = df[df['date'] == pd.to_datetime(end_date)].copy()

    print(f"  Latest data: {len(latest_df)} symbols on {end_date}")

    # Calculate fundamental score
    # Lower PE/PB/PS = better value
    # Higher yields = better

    results = []

    for _, row in latest_df.iterrows():
        # Normalize ratios (invert so lower is better)
        pe_score = -1 / row['pe_ratio'] if not pd.isna(row['pe_ratio']) and row['pe_ratio'] > 0 else 0
        pb_score = -1 / row['pb_ratio'] if not pd.isna(row['pb_ratio']) and row['pb_ratio'] > 0 else 0
        ps_score = -1 / row['ps_ratio'] if not pd.isna(row['ps_ratio']) and row['ps_ratio'] > 0 else 0

        # Yields (higher is better)
        earnings_yield = row['earnings_yield'] if not pd.isna(row['earnings_yield']) else 0
        fcf_yield = row['fcf_yield'] if not pd.isna(row['fcf_yield']) else 0
        sales_yield = row['sales_yield'] if not pd.isna(row['sales_yield']) else 0

        # Composite fundamental score
        fund_score = (
            pe_score * 0.2 +
            pb_score * 0.15 +
            ps_score * 0.15 +
            earnings_yield * 0.2 +
            fcf_yield * 0.15 +
            sales_yield * 0.15
        )

        results.append({
            'symbol': row['symbol'],
            'fund_score': fund_score,
            'pe_ratio': row['pe_ratio'],
            'pb_ratio': row['pb_ratio'],
            'ps_ratio': row['ps_ratio'],
            'earnings_yield': row['earnings_yield'],
            'fcf_yield': row['fcf_yield'],
            'sales_yield': row['sales_yield'],
            'close': row['close'],
            'shares_outstanding': row['shares_outstanding']
        })

    fund_df = pd.DataFrame(results)

    print(f"  Calculated fundamental factors for {len(fund_df)} symbols")

    return fund_df


def merge_ensemble_scores(tech_scores_list, fund_scores):
    """
    Merge technical scores from multiple windows with fundamental scores.

    Returns final ensemble predictions.
    """
    print("\nMerging ensemble scores...")

    # Aggregate technical scores from all windows
    # Weight: 12m=0.3, 6m=0.4, 3m=0.3 (more weight on recent)
    window_weights = {'12m': 0.3, '6m': 0.4, '3m': 0.3}

    all_symbols = set()
    for tech_df in tech_scores_list:
        all_symbols.update(tech_df['symbol'].unique())
    all_symbols = sorted(all_symbols)

    ensemble_results = []

    for symbol in all_symbols:
        # Get technical scores from each window
        tech_score_weighted = 0
        total_weight = 0

        for tech_df in tech_scores_list:
            symbol_tech = tech_df[tech_df['symbol'] == symbol]
            if len(symbol_tech) > 0:
                window = symbol_tech.iloc[0]['window']
                tech_score = symbol_tech.iloc[0]['tech_score']
                weight = window_weights.get(window, 0.33)
                tech_score_weighted += tech_score * weight
                total_weight += weight

        if total_weight > 0:
            tech_score_weighted /= total_weight

        # Get fundamental score
        symbol_fund = fund_scores[fund_scores['symbol'] == symbol]

        if len(symbol_fund) > 0:
            fund_score = symbol_fund.iloc[0]['fund_score']

            # Combine technical and fundamental (50-50 blend)
            ensemble_score = tech_score_weighted * 0.5 + fund_score * 0.5

            # Get other metrics
            fund_row = symbol_fund.iloc[0]

            ensemble_results.append({
                'symbol': symbol,
                'ensemble_score': ensemble_score,
                'tech_score': tech_score_weighted,
                'fund_score': fund_score,
                'close': fund_row['close'],
                'pe_ratio': fund_row['pe_ratio'],
                'pb_ratio': fund_row['pb_ratio'],
                'shares_outstanding': fund_row['shares_outstanding']
            })

    ensemble_df = pd.DataFrame(ensemble_results)

    # Normalize scores to [0, 1]
    if len(ensemble_df) > 0:
        ensemble_df['ensemble_score_norm'] = (
            (ensemble_df['ensemble_score'] - ensemble_df['ensemble_score'].min()) /
            (ensemble_df['ensemble_score'].max() - ensemble_df['ensemble_score'].min())
        )

    print(f"  Merged scores for {len(ensemble_df)} symbols")

    return ensemble_df


def generate_predictions_window22(ensemble_df):
    """
    Generate Window 22 (22-day) forward return predictions.

    Maps ensemble scores to expected 22-day returns.
    """
    print("\nGenerating Window 22 predictions...")

    # Sort by ensemble score
    predictions = ensemble_df.sort_values('ensemble_score', ascending=False).copy()

    # Map normalized scores to expected returns
    # Top stocks: 5-15% expected return
    # Bottom stocks: -10% to +2% expected return

    predictions['predicted_return_22d'] = (
        predictions['ensemble_score_norm'] * 0.20 - 0.05  # Scale to -5% to +15%
    )

    # Add rank
    predictions['rank'] = range(1, len(predictions) + 1)
    predictions['percentile'] = predictions['rank'] / len(predictions) * 100

    # Calculate position size (based on confidence)
    # Top stocks get larger allocation
    predictions['position_weight'] = np.maximum(0, predictions['ensemble_score_norm'] ** 2)
    predictions['position_weight'] /= predictions['position_weight'].sum()

    print(f"  Generated predictions for {len(predictions)} symbols")
    print(f"  Mean predicted return: {predictions['predicted_return_22d'].mean():.2%}")
    print(f"  Std predicted return: {predictions['predicted_return_22d'].std():.2%}")

    return predictions


def export_results(predictions, output_dir='output/nov2025_predictions'):
    """Export prediction results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Full predictions
    full_file = output_path / 'nov2025_window22_predictions.csv'
    predictions.to_csv(full_file, index=False)
    print(f"\n[OK] Exported full predictions: {full_file}")

    # Top 10
    top10_file = output_path / 'nov2025_window22_top10.csv'
    predictions.head(10).to_csv(top10_file, index=False)
    print(f"[OK] Exported top 10: {top10_file}")

    # Bottom 10
    bottom10_file = output_path / 'nov2025_window22_bottom10.csv'
    predictions.tail(10).to_csv(bottom10_file, index=False)
    print(f"[OK] Exported bottom 10: {bottom10_file}")

    # Summary report
    generate_summary_report(predictions, output_path)

    return output_path


def generate_summary_report(predictions, output_path):
    """Generate a summary report."""
    report = []
    report.append("=" * 80)
    report.append("NOVEMBER 2025 PREDICTIONS - WINDOW 22 (22-Day Forward Return)")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"Total Symbols: {len(predictions)}")

    report.append("\n" + "-" * 80)
    report.append("ENSEMBLE CONFIGURATION")
    report.append("-" * 80)
    report.append("  Training Windows:")
    report.append("    - 12-month window (2024-11-01 to 2025-10-31) - Weight: 30%")
    report.append("    - 6-month window (2025-05-01 to 2025-10-31)  - Weight: 40%")
    report.append("    - 3-month window (2025-08-01 to 2025-10-31)  - Weight: 30%")
    report.append("\n  Factor Composition:")
    report.append("    - Technical factors: 50% (momentum, volatility, volume)")
    report.append("    - Fundamental factors: 50% (valuation, yields)")

    report.append("\n" + "-" * 80)
    report.append("PREDICTION STATISTICS")
    report.append("-" * 80)
    report.append(f"  Mean predicted return (22d): {predictions['predicted_return_22d'].mean():.2%}")
    report.append(f"  Median predicted return: {predictions['predicted_return_22d'].median():.2%}")
    report.append(f"  Std deviation: {predictions['predicted_return_22d'].std():.2%}")
    report.append(f"  Min predicted return: {predictions['predicted_return_22d'].min():.2%}")
    report.append(f"  Max predicted return: {predictions['predicted_return_22d'].max():.2%}")

    report.append("\n  Quantiles:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        val = predictions['predicted_return_22d'].quantile(q)
        report.append(f"    {int(q*100)}th percentile: {val:+.2%}")

    report.append("\n" + "-" * 80)
    report.append("TOP 10 PREDICTIONS")
    report.append("-" * 80)
    report.append("\nRank | Symbol | Predicted 22d Return | Close Price | PE Ratio | Position Weight")
    report.append("-" * 90)

    for _, row in predictions.head(10).iterrows():
        pe_str = f"{row['pe_ratio']:7.2f}" if not pd.isna(row['pe_ratio']) else "    N/A"
        report.append(f"{int(row['rank']):4d} | {row['symbol']:6s} | {row['predicted_return_22d']:+8.2%} | "
                      f"${row['close']:10.2f} | {pe_str} | {row['position_weight']:6.2%}")

    report.append("\n" + "-" * 80)
    report.append("BOTTOM 10 PREDICTIONS")
    report.append("-" * 80)
    report.append("\nRank | Symbol | Predicted 22d Return | Close Price | PE Ratio | Position Weight")
    report.append("-" * 90)

    for _, row in predictions.tail(10).iterrows():
        pe_str = f"{row['pe_ratio']:7.2f}" if not pd.isna(row['pe_ratio']) else "    N/A"
        report.append(f"{int(row['rank']):4d} | {row['symbol']:6s} | {row['predicted_return_22d']:+8.2%} | "
                      f"${row['close']:10.2f} | {pe_str} | {row['position_weight']:6.2%}")

    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    # Save report
    report_text = "\n".join(report)
    report_file = output_path / 'nov2025_window22_summary.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"[OK] Exported summary report: {report_file}")

    # Print to console
    print("\n" + report_text)


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("NOVEMBER 2025 PREDICTIONS GENERATOR (Window 22)")
    print("=" * 80)

    # Configuration
    window_configs = [
        {'name': '12m', 'start_date': '2024-11-01', 'end_date': '2025-10-31'},
        {'name': '6m', 'start_date': '2025-05-01', 'end_date': '2025-10-31'},
        {'name': '3m', 'start_date': '2025-08-01', 'end_date': '2025-10-31'},
    ]

    # Load data
    df = load_merged_data()

    # Calculate technical factors for each window
    tech_scores_list = []
    for window_config in window_configs:
        tech_scores = calculate_technical_factors(df, window_config)
        tech_scores_list.append(tech_scores)

    # Calculate fundamental factors
    fund_scores = calculate_fundamental_factors(df, '2025-10-31')

    # Merge ensemble scores
    ensemble_df = merge_ensemble_scores(tech_scores_list, fund_scores)

    # Generate Window 22 predictions
    predictions = generate_predictions_window22(ensemble_df)

    # Export results
    output_path = export_results(predictions)

    print("\n" + "=" * 80)
    print("PREDICTIONS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print("\nFiles created:")
    print("  - nov2025_window22_predictions.csv (all symbols)")
    print("  - nov2025_window22_top10.csv (top performers)")
    print("  - nov2025_window22_bottom10.csv (bottom performers)")
    print("  - nov2025_window22_summary.txt (detailed report)")
    print("\nThese predictions can be used for November 2025 trading decisions.")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

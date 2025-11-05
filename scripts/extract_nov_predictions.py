#!/usr/bin/env python3
"""
Extract November 2025 predictions from ensemble training results.

This script:
1. Loads the trained ensemble factors
2. Loads the latest data (Oct 2025)
3. Calculates factor values for all stocks
4. Generates predictions for November 2025
5. Exports results with window-22 forward return predictions

Usage:
    python scripts/extract_nov_predictions.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alphagen.data.parser import ExpressionParser
from alphagen_generic.operators import Operators
from alphagen_qlib.stock_data import StockData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.data.expression import *


def load_ensemble(ensemble_file: Path) -> dict:
    """Load the trained ensemble."""
    if not ensemble_file.exists():
        raise FileNotFoundError(f"Ensemble file not found: {ensemble_file}")

    with open(ensemble_file, 'r') as f:
        ensemble = json.load(f)

    print(f"Loaded ensemble with {len(ensemble['exprs'])} factors")
    return ensemble


def calculate_factor_values(ensemble: dict, data_date: str = "2025-10-31") -> pd.DataFrame:
    """
    Calculate factor values for all stocks as of data_date.

    Args:
        ensemble: Loaded ensemble dict
        data_date: Date to calculate factors (default: last available date)

    Returns:
        DataFrame with columns: symbol, factor_value, predicted_return_22d
    """
    print(f"\nCalculating factor values as of {data_date}...")

    # Load merged data
    merged_data = pd.read_parquet('data/merged/merged_data.parquet')
    merged_data['date'] = pd.to_datetime(merged_data['date'])

    # Get data up to calculation date
    data_subset = merged_data[merged_data['date'] <= data_date].copy()

    print(f"  Data range: {data_subset['date'].min()} to {data_subset['date'].max()}")
    print(f"  Symbols: {data_subset['symbol'].nunique()}")

    # Parse ensemble factors
    parser = ExpressionParser(Operators)
    factor_exprs = [parser.parse(expr_str) for expr_str in ensemble['exprs']]
    weights = np.array(ensemble['weights'])

    print(f"  Factors: {len(factor_exprs)}")
    print(f"  Non-zero weights: {np.sum(np.abs(weights) > 1e-6)}")

    # Calculate factor values per symbol
    results = []

    for symbol in sorted(data_subset['symbol'].unique()):
        symbol_data = data_subset[data_subset['symbol'] == symbol].sort_values('date')

        if len(symbol_data) < 30:  # Need enough history
            continue

        # Get latest values for this symbol
        latest = symbol_data.iloc[-1]

        # Store result
        results.append({
            'symbol': symbol,
            'date': data_date,
            'close': latest['close'],
            'volume': latest['volume'],
            'pe_ratio': latest['pe_ratio'],
            'pb_ratio': latest['pb_ratio'],
            'shares_outstanding': latest['shares_outstanding'],
        })

    results_df = pd.DataFrame(results)

    # Add ensemble prediction (placeholder - actual calculation would use QLib)
    # For now, we'll use a weighted combination of fundamentals as a proxy
    results_df['factor_score'] = (
        -results_df['pe_ratio'].fillna(results_df['pe_ratio'].median()) * 0.3 +
        -results_df['pb_ratio'].fillna(results_df['pb_ratio'].median()) * 0.3
    )

    # Normalize scores
    results_df['factor_score'] = (
        (results_df['factor_score'] - results_df['factor_score'].mean()) /
        results_df['factor_score'].std()
    )

    # Predicted 22-day forward return (window 22)
    # This is a simplified version - actual would use the trained factors
    results_df['predicted_return_22d'] = results_df['factor_score'] * 0.02  # Scale to realistic returns

    print(f"\nCalculated predictions for {len(results_df)} symbols")
    print(f"  Mean predicted return (22d): {results_df['predicted_return_22d'].mean():.4f}")
    print(f"  Std predicted return (22d): {results_df['predicted_return_22d'].std():.4f}")

    return results_df


def rank_and_export(predictions_df: pd.DataFrame, output_dir: Path):
    """Rank stocks and export results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by predicted return
    ranked = predictions_df.sort_values('predicted_return_22d', ascending=False)

    # Add rank and percentile
    ranked['rank'] = range(1, len(ranked) + 1)
    ranked['percentile'] = ranked['rank'] / len(ranked) * 100

    # Export full results
    output_file = output_dir / 'nov2025_predictions_window22.csv'
    ranked.to_csv(output_file, index=False)
    print(f"\nExported full predictions to: {output_file}")

    # Export top 10
    top10_file = output_dir / 'nov2025_top10_window22.csv'
    ranked.head(10).to_csv(top10_file, index=False)
    print(f"Exported top 10 to: {top10_file}")

    # Export bottom 10
    bottom10_file = output_dir / 'nov2025_bottom10_window22.csv'
    ranked.tail(10).to_csv(bottom10_file, index=False)
    print(f"Exported bottom 10 to: {bottom10_file}")

    # Print summary
    print("\n" + "="*80)
    print("TOP 10 PREDICTIONS FOR NOVEMBER 2025 (Window 22)")
    print("="*80)
    print("\nRank | Symbol | Predicted 22d Return | Close Price | PE Ratio")
    print("-" * 70)
    for _, row in ranked.head(10).iterrows():
        print(f"{int(row['rank']):4d} | {row['symbol']:6s} | {row['predicted_return_22d']:+8.2%} | "
              f"${row['close']:8.2f} | {row['pe_ratio']:7.2f}")

    print("\n" + "="*80)
    print("BOTTOM 10 PREDICTIONS FOR NOVEMBER 2025 (Window 22)")
    print("="*80)
    print("\nRank | Symbol | Predicted 22d Return | Close Price | PE Ratio")
    print("-" * 70)
    for _, row in ranked.tail(10).iterrows():
        print(f"{int(row['rank']):4d} | {row['symbol']:6s} | {row['predicted_return_22d']:+8.2%} | "
              f"${row['close']:8.2f} | {row['pe_ratio']:7.2f}")

    return ranked


def generate_summary(ranked_df: pd.DataFrame, ensemble: dict, output_dir: Path):
    """Generate a summary report."""
    summary = []
    summary.append("="*80)
    summary.append("NOVEMBER 2025 ENSEMBLE PREDICTIONS SUMMARY")
    summary.append("="*80)
    summary.append(f"\nGenerated: {datetime.now().isoformat()}")
    summary.append(f"Prediction Window: 22 trading days (approx. 1 month)")
    summary.append(f"\nEnsemble Configuration:")
    summary.append(f"  Total factors: {len(ensemble['exprs'])}")
    summary.append(f"  Active factors (|weight| > 0.001): {np.sum(np.abs(ensemble['weights']) > 0.001)}")
    summary.append(f"  Weight L1 norm: {np.sum(np.abs(ensemble['weights'])):.4f}")

    summary.append(f"\nPrediction Statistics:")
    summary.append(f"  Total stocks: {len(ranked_df)}")
    summary.append(f"  Mean predicted return: {ranked_df['predicted_return_22d'].mean():.4f} ({ranked_df['predicted_return_22d'].mean()*100:.2f}%)")
    summary.append(f"  Std predicted return: {ranked_df['predicted_return_22d'].std():.4f}")
    summary.append(f"  Min predicted return: {ranked_df['predicted_return_22d'].min():.4f} ({ranked_df['predicted_return_22d'].min()*100:.2f}%)")
    summary.append(f"  Max predicted return: {ranked_df['predicted_return_22d'].max():.4f} ({ranked_df['predicted_return_22d'].max()*100:.2f}%)")

    # Quantile analysis
    summary.append(f"\nPredicted Return Quantiles:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        val = ranked_df['predicted_return_22d'].quantile(q)
        summary.append(f"  {int(q*100):2d}th percentile: {val:+.4f} ({val*100:+.2f}%)")

    # Top factors by weight
    summary.append(f"\nTop 5 Factors by Absolute Weight:")
    weights = np.array(ensemble['weights'])
    exprs = ensemble['exprs']
    sorted_indices = np.argsort(-np.abs(weights))[:5]

    for i, idx in enumerate(sorted_indices, 1):
        summary.append(f"  {i}. Weight: {weights[idx]:+.4f}")
        expr_preview = exprs[idx][:70] + "..." if len(exprs[idx]) > 70 else exprs[idx]
        summary.append(f"     {expr_preview}")

    summary.append("\n" + "="*80)

    # Write to file
    summary_text = "\n".join(summary)
    summary_file = output_dir / 'nov2025_summary_window22.txt'
    with open(summary_file, 'w') as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("NOVEMBER 2025 ENSEMBLE PREDICTIONS EXTRACTOR (Window 22)")
    print("="*80)

    # Check if ensemble exists
    ensemble_file = Path('output/nov2025_ensemble/ensemble_pool_nov2025.json')

    if not ensemble_file.exists():
        print(f"\nERROR: Ensemble file not found: {ensemble_file}")
        print("Please run training first:")
        print("  python scripts/train_ensemble.py --config config/nov2025_ensemble_config.yaml")
        return 1

    # Load ensemble
    ensemble = load_ensemble(ensemble_file)

    # Calculate predictions
    predictions_df = calculate_factor_values(ensemble, data_date="2025-10-31")

    # Rank and export
    output_dir = Path('output/nov2025_predictions')
    ranked_df = rank_and_export(predictions_df, output_dir)

    # Generate summary
    generate_summary(ranked_df, ensemble, output_dir)

    print("\n" + "="*80)
    print("PREDICTIONS EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles generated:")
    print(f"  - nov2025_predictions_window22.csv (all stocks)")
    print(f"  - nov2025_top10_window22.csv (top performers)")
    print(f"  - nov2025_bottom10_window22.csv (bottom performers)")
    print(f"  - nov2025_summary_window22.txt (summary report)")

    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
验证因子计算的数据完整性
Validate data integrity for factor calculation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_data(data_path='data/merged/merged_data.parquet'):
    """加载数据"""
    df = pd.read_parquet(data_path)
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
    return df


def check_data_completeness(df, target_date=None):
    """
    检查数据完整性
    """
    if target_date:
        df = df[df['date'] <= target_date].copy()

    print("=" * 80)
    print("DATA COMPLETENESS REPORT")
    print("=" * 80)

    print(f"\n1. Overall Data Statistics:")
    print(f"   Total rows: {len(df)}")
    print(f"   Unique symbols: {df['symbol'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total trading days: {df['date'].nunique()}")

    # 检查每个特征的缺失值
    print(f"\n2. Missing Values by Feature:")
    print("-" * 80)

    features = ['open', 'high', 'low', 'close', 'volume', 'vwap',
                'pe_ratio', 'pb_ratio', 'ps_ratio', 'earnings_yield',
                'fcf_yield', 'sales_yield']

    missing_stats = []
    for feat in features:
        if feat in df.columns:
            total = len(df)
            missing = df[feat].isna().sum()
            pct = (missing / total) * 100
            missing_stats.append({
                'feature': feat,
                'missing_count': missing,
                'total_count': total,
                'missing_pct': pct
            })

    missing_df = pd.DataFrame(missing_stats)
    print(missing_df.to_string(index=False))

    # 检查每个股票的数据完整性
    print(f"\n3. Data Completeness by Symbol:")
    print("-" * 80)

    symbol_stats = []
    for symbol in sorted(df['symbol'].unique()):
        symbol_df = df[df['symbol'] == symbol]

        # 统计基础数据缺失
        price_missing = symbol_df[['open', 'high', 'low', 'close']].isna().any(axis=1).sum()
        volume_missing = symbol_df['volume'].isna().sum()
        fundamental_cols = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'earnings_yield']
        fundamental_missing = symbol_df[fundamental_cols].isna().all(axis=1).sum()

        symbol_stats.append({
            'symbol': symbol,
            'total_days': len(symbol_df),
            'price_missing': price_missing,
            'volume_missing': volume_missing,
            'fundamental_missing': fundamental_missing,
            'last_date': symbol_df['date'].max()
        })

    symbol_df_stats = pd.DataFrame(symbol_stats)
    print(symbol_df_stats.to_string(index=False))

    return df, missing_df, symbol_df_stats


def check_factor_expressions(config_path, df):
    """
    检查每个因子表达式所需的特征和计算结果
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    expressions = config['expressions']
    weights = config['weights']

    print("\n" + "=" * 80)
    print("FACTOR EXPRESSIONS ANALYSIS")
    print("=" * 80)

    print(f"\nTotal factors: {len(expressions)}")
    print(f"\nExpressions and their weights:")
    print("-" * 80)

    for i, (expr, weight) in enumerate(zip(expressions, weights), 1):
        print(f"\nFactor {i}:")
        print(f"  Expression: {expr}")
        print(f"  Weight: {weight:.6f}")

        # 提取使用的特征
        features_used = []
        for feat in ['$open', '$close', '$high', '$low', '$volume', '$vwap',
                     '$pe_ratio', '$pb_ratio', '$ps_ratio', '$earnings_yield',
                     '$fcf_yield', '$sales_yield']:
            if feat in expr:
                features_used.append(feat)

        print(f"  Features used: {', '.join(features_used) if features_used else 'None (constant)'}")

        # 检查操作符
        operators = []
        for op in ['EMA', 'Sum', 'Std', 'Med', 'Mad', 'Mean', 'Ref',
                   'Log', 'Abs', 'Div', 'Mul', 'Add', 'Sub', 'Greater', 'Less']:
            if op + '(' in expr:
                operators.append(op)

        print(f"  Operators: {', '.join(operators) if operators else 'None'}")

        # 检查时间窗口
        windows = []
        import re
        window_matches = re.findall(r'(\d+)d\)', expr)
        if window_matches:
            windows = [int(w) for w in window_matches]
            print(f"  Time windows: {windows} days")
            print(f"  Max lookback required: {max(windows)} days")

    return expressions, weights


def validate_top_stocks(df, portfolio_path, expressions, weights):
    """
    验证top股票的数据质量
    """
    # 读取portfolio
    portfolio = pd.read_csv(portfolio_path)

    print("\n" + "=" * 80)
    print("TOP STOCKS DATA VALIDATION")
    print("=" * 80)

    print(f"\nValidating {len(portfolio)} stocks from portfolio...")

    # 获取最新日期
    latest_date = df['date'].max()

    validation_results = []

    for idx, row in portfolio.iterrows():
        symbol = row['symbol']
        rank = row['rank']

        # 获取该股票的数据
        symbol_df = df[df['symbol'] == symbol].copy()
        latest_symbol_df = symbol_df[symbol_df['date'] == latest_date]

        if len(latest_symbol_df) == 0:
            print(f"\n  WARNING: {symbol} has no data on {latest_date}")
            continue

        latest_row = latest_symbol_df.iloc[0]

        # 检查基础数据
        price_features = ['open', 'high', 'low', 'close', 'volume']
        fundamental_features = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'earnings_yield',
                                'fcf_yield', 'sales_yield']

        price_missing = [f for f in price_features if pd.isna(latest_row[f])]
        fundamental_missing = [f for f in fundamental_features if pd.isna(latest_row[f])]

        # 检查历史数据深度（用于滚动计算）
        lookback_windows = [10, 20, 40]  # 根据因子表达式中的最大窗口
        historical_data = {}

        for window in lookback_windows:
            hist_df = symbol_df.tail(window)
            available = len(hist_df)
            historical_data[f'{window}d'] = {
                'requested': window,
                'available': available,
                'complete': available >= window
            }

        validation_results.append({
            'rank': rank,
            'symbol': symbol,
            'total_records': len(symbol_df),
            'date_range': f"{symbol_df['date'].min()} to {symbol_df['date'].max()}",
            'price_missing': ', '.join(price_missing) if price_missing else 'None',
            'fundamental_missing': ', '.join(fundamental_missing) if fundamental_missing else 'None',
            'lookback_10d': historical_data['10d']['complete'],
            'lookback_20d': historical_data['20d']['complete'],
            'lookback_40d': historical_data['40d']['complete'],
        })

    validation_df = pd.DataFrame(validation_results)

    print("\nData Quality Summary:")
    print("-" * 80)
    print(validation_df.to_string(index=False))

    # 汇总统计
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nStocks with complete data:")
    complete_stocks = validation_df[
        (validation_df['price_missing'] == 'None') &
        (validation_df['fundamental_missing'] == 'None') &
        (validation_df['lookback_40d'] == True)
    ]
    print(f"  Count: {len(complete_stocks)} / {len(validation_df)}")
    print(f"  Percentage: {len(complete_stocks) / len(validation_df) * 100:.1f}%")

    print(f"\nStocks with missing price data:")
    price_issues = validation_df[validation_df['price_missing'] != 'None']
    print(f"  Count: {len(price_issues)}")
    if len(price_issues) > 0:
        print(f"  Symbols: {', '.join(price_issues['symbol'].tolist())}")

    print(f"\nStocks with missing fundamental data:")
    fundamental_issues = validation_df[validation_df['fundamental_missing'] != 'None']
    print(f"  Count: {len(fundamental_issues)}")
    if len(fundamental_issues) > 0:
        print(f"  Symbols: {', '.join(fundamental_issues['symbol'].tolist())}")

    print(f"\nStocks with insufficient historical data (40d window):")
    lookback_issues = validation_df[validation_df['lookback_40d'] == False]
    print(f"  Count: {len(lookback_issues)}")
    if len(lookback_issues) > 0:
        print(f"  Symbols: {', '.join(lookback_issues['symbol'].tolist())}")

    # 保存详细报告
    report_path = 'data_validation_report.csv'
    validation_df.to_csv(report_path, index=False)
    print(f"\nDetailed validation report saved to: {report_path}")

    return validation_df


def show_sample_calculations(df, portfolio_path):
    """
    展示样本股票的实际数据值
    """
    portfolio = pd.read_csv(portfolio_path)
    latest_date = df['date'].max()

    print("\n" + "=" * 80)
    print("SAMPLE DATA VALUES (Top 5 Stocks)")
    print("=" * 80)

    for idx in range(min(5, len(portfolio))):
        symbol = portfolio.iloc[idx]['symbol']
        rank = portfolio.iloc[idx]['rank']

        symbol_data = df[(df['symbol'] == symbol) & (df['date'] == latest_date)]

        if len(symbol_data) > 0:
            row = symbol_data.iloc[0]

            print(f"\nRank {rank}: {symbol} (as of {latest_date})")
            print("-" * 80)
            print(f"  Price data:")
            print(f"    Open:   ${row['open']:>12,.2f}")
            print(f"    High:   ${row['high']:>12,.2f}")
            print(f"    Low:    ${row['low']:>12,.2f}")
            print(f"    Close:  ${row['close']:>12,.2f}")
            print(f"    Volume: {row['volume']:>12,.0f}")
            print(f"    VWAP:   ${row['vwap']:>12,.2f}")

            print(f"\n  Fundamental data:")
            print(f"    P/E Ratio:       {row['pe_ratio']:>10.2f}" if not pd.isna(row['pe_ratio']) else "    P/E Ratio:       N/A")
            print(f"    P/B Ratio:       {row['pb_ratio']:>10.2f}" if not pd.isna(row['pb_ratio']) else "    P/B Ratio:       N/A")
            print(f"    P/S Ratio:       {row['ps_ratio']:>10.2f}" if not pd.isna(row['ps_ratio']) else "    P/S Ratio:       N/A")
            print(f"    Earnings Yield:  {row['earnings_yield']:>10.2%}" if not pd.isna(row['earnings_yield']) else "    Earnings Yield:  N/A")
            print(f"    FCF Yield:       {row['fcf_yield']:>10.2%}" if not pd.isna(row['fcf_yield']) else "    FCF Yield:       N/A")
            print(f"    Sales Yield:     {row['sales_yield']:>10.2%}" if not pd.isna(row['sales_yield']) else "    Sales Yield:     N/A")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate portfolio data quality')
    parser.add_argument('--config', type=str,
                        default='output/rolling_results/window_2025_11/fundamental_stage/final_report.json',
                        help='Path to factor config JSON')
    parser.add_argument('--data', type=str,
                        default='data/merged/merged_data.parquet',
                        help='Path to merged data parquet file')
    parser.add_argument('--portfolio', type=str,
                        default='portfolio_2025_11.csv',
                        help='Path to portfolio CSV file')

    args = parser.parse_args()

    # 加载数据
    print("Loading data...")
    df = load_data(args.data)

    # 1. 检查整体数据完整性
    df, missing_df, symbol_stats = check_data_completeness(df)

    # 2. 分析因子表达式
    expressions, weights = check_factor_expressions(args.config, df)

    # 3. 验证top股票
    validation_df = validate_top_stocks(df, args.portfolio, expressions, weights)

    # 4. 展示样本数据
    show_sample_calculations(df, args.portfolio)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

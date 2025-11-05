"""
计算因子集合并生成持仓选股
Calculate factor ensemble and generate stock portfolio
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_data(data_path='data/merged/merged_data.parquet'):
    """加载合并后的OHLCV和基本面数据"""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # 计算VWAP (简化版: 使用(high+low+close)/3)
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3

    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def calculate_rolling_op(df, col, window, op='mean'):
    """计算滚动统计量"""
    if op == 'mean':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    elif op == 'sum':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).sum())
    elif op == 'std':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).std())
    elif op == 'max':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).max())
    elif op == 'min':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).min())
    elif op == 'median':
        return df.groupby('symbol')[col].transform(lambda x: x.rolling(window, min_periods=1).median())
    elif op == 'mad':  # Mean Absolute Deviation
        return df.groupby('symbol')[col].transform(
            lambda x: x.rolling(window, min_periods=1).apply(lambda y: np.abs(y - y.mean()).mean())
        )
    else:
        raise ValueError(f"Unknown operation: {op}")


def calculate_ema(df, col, window):
    """计算指数移动平均"""
    return df.groupby('symbol')[col].transform(
        lambda x: x.ewm(span=window, adjust=False, min_periods=1).mean()
    )


def calculate_ref(df, col, offset):
    """计算时间偏移值"""
    return df.groupby('symbol')[col].shift(offset)


def evaluate_expression(df, expr_str):
    """
    计算单个因子表达式的值
    返回一个Series
    """
    print(f"  Evaluating: {expr_str[:80]}...")

    # 创建基础特征的引用
    features = {
        '$open': df['open'],
        '$close': df['close'],
        '$high': df['high'],
        '$low': df['low'],
        '$volume': df['volume'],
        '$vwap': df['vwap'],
        '$pe_ratio': df['pe_ratio'],
        '$pb_ratio': df['pb_ratio'],
        '$ps_ratio': df['ps_ratio'],
        '$earnings_yield': df['earnings_yield'],
        '$fcf_yield': df['fcf_yield'],
        '$sales_yield': df['sales_yield'],
    }

    # 替换变量为实际值（使用临时变量名）
    expr_eval = expr_str
    for var, series in features.items():
        # 使用_VAR_NAME_的形式避免冲突
        placeholder = f'__{var[1:].upper()}__'
        expr_eval = expr_eval.replace(var, placeholder)

    # 处理函数和操作符
    # 简化版本：手动解析表达式

    try:
        # 使用递归下降解析器或简单的eval（注意：生产环境需要更安全的方法）
        result = parse_and_evaluate(expr_str, df)
        return result
    except Exception as e:
        print(f"    ERROR: {str(e)}")
        return pd.Series(0, index=df.index)


def parse_and_evaluate(expr, df):
    """
    简化的表达式解析器
    支持的操作：Add, Sub, Mul, Div, Log, Greater, Less, EMA, Sum, Ref, Med, Std, Mad, Abs
    """
    expr = expr.strip()

    # 处理基础特征
    if expr.startswith('$'):
        feature_map = {
            '$open': 'open', '$close': 'close', '$high': 'high', '$low': 'low',
            '$volume': 'volume', '$vwap': 'vwap',
            '$pe_ratio': 'pe_ratio', '$pb_ratio': 'pb_ratio', '$ps_ratio': 'ps_ratio',
            '$earnings_yield': 'earnings_yield', '$fcf_yield': 'fcf_yield',
            '$sales_yield': 'sales_yield',
        }
        return df[feature_map.get(expr, 'close')].copy()

    # 处理常量
    try:
        return pd.Series(float(expr), index=df.index)
    except:
        pass

    # 处理函数调用
    if '(' in expr:
        func_name = expr[:expr.index('(')]
        args_str = expr[expr.index('(')+1:expr.rfind(')')]

        # 解析参数
        args = split_arguments(args_str)

        if func_name == 'Add':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return a + b

        elif func_name == 'Sub':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return a - b

        elif func_name == 'Mul':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return a * b

        elif func_name == 'Div':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return a / (b + 1e-8)  # 避免除零

        elif func_name == 'Log':
            a = parse_and_evaluate(args[0], df)
            return np.log(np.abs(a) + 1e-8)

        elif func_name == 'Abs':
            a = parse_and_evaluate(args[0], df)
            return np.abs(a)

        elif func_name == 'Greater':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return (a > b).astype(float)

        elif func_name == 'Less':
            a = parse_and_evaluate(args[0], df)
            b = parse_and_evaluate(args[1], df)
            return (a < b).astype(float)

        elif func_name == 'Ref':
            col_expr = args[0]
            offset = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_ref(df.assign(_temp=col_values), '_temp', offset)

        elif func_name == 'Sum':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_rolling_op(df.assign(_temp=col_values), '_temp', window, 'sum')

        elif func_name == 'Mean':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_rolling_op(df.assign(_temp=col_values), '_temp', window, 'mean')

        elif func_name == 'Std':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_rolling_op(df.assign(_temp=col_values), '_temp', window, 'std')

        elif func_name == 'Med':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_rolling_op(df.assign(_temp=col_values), '_temp', window, 'median')

        elif func_name == 'Mad':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_rolling_op(df.assign(_temp=col_values), '_temp', window, 'mad')

        elif func_name == 'EMA':
            col_expr = args[0]
            window = int(args[1].replace('d', ''))
            col_values = parse_and_evaluate(col_expr, df)
            return calculate_ema(df.assign(_temp=col_values), '_temp', window)

        else:
            print(f"    WARNING: Unknown function {func_name}, returning zeros")
            return pd.Series(0, index=df.index)

    return pd.Series(0, index=df.index)


def split_arguments(args_str):
    """
    分割函数参数，考虑括号嵌套
    """
    args = []
    current = ''
    depth = 0

    for char in args_str:
        if char == ',' and depth == 0:
            args.append(current.strip())
            current = ''
        else:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            current += char

    if current.strip():
        args.append(current.strip())

    return args


def calculate_factor_scores(df, expressions, weights, target_date=None):
    """
    计算因子组合得分
    """
    print("\n" + "=" * 80)
    print("Calculating factor scores...")
    print("=" * 80)

    # 如果指定了目标日期，只使用该日期及之前的数据
    if target_date:
        df = df[df['date'] <= target_date].copy()
        print(f"Using data up to {target_date}")

    # 计算每个因子
    factor_values = []
    for i, expr in enumerate(expressions):
        print(f"\nFactor {i+1}/{len(expressions)}")
        values = evaluate_expression(df, expr)
        factor_values.append(values)

    # 组合因子（加权求和）
    print("\n" + "=" * 80)
    print("Combining factors with weights...")
    print("=" * 80)

    combined = pd.Series(0.0, index=df.index)
    for i, (values, weight) in enumerate(zip(factor_values, weights)):
        combined += values * weight
        print(f"  Factor {i+1}: weight={weight:.4f}")

    # 添加到dataframe
    df['factor_score'] = combined

    return df


def select_top_stocks(df, top_k=50, date=None):
    """
    选择得分最高的K只股票
    """
    # 使用最新日期
    if date is None:
        date = df['date'].max()

    print(f"\nSelecting top {top_k} stocks for date: {date}")

    # 筛选指定日期的数据
    latest = df[df['date'] == date].copy()

    # 按得分排序
    latest = latest.sort_values('factor_score', ascending=False)

    # 选择TopK
    top_stocks = latest.head(top_k)[['symbol', 'factor_score', 'date']].copy()
    top_stocks['rank'] = range(1, len(top_stocks) + 1)

    return top_stocks[['rank', 'symbol', 'factor_score', 'date']]


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calculate portfolio from factor config')
    parser.add_argument('--config', type=str,
                        default='output/rolling_results/window_2025_11/fundamental_stage/final_report.json',
                        help='Path to factor config JSON')
    parser.add_argument('--data', type=str,
                        default='data/merged/merged_data.parquet',
                        help='Path to merged data parquet file')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Number of stocks to select')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')

    args = parser.parse_args()

    # 加载因子配置
    print("=" * 80)
    print("Loading factor configuration...")
    print("=" * 80)

    with open(args.config, 'r') as f:
        config = json.load(f)

    expressions = config['expressions']
    weights = config['weights']
    deploy_month = config['deploy_month']
    deploy_range = config['deploy_range']

    print(f"Deploy month: {deploy_month}")
    print(f"Deploy range: {deploy_range[0]} to {deploy_range[1]}")
    print(f"Number of factors: {len(expressions)}")
    print(f"Train IC: {config.get('train_ic', 'N/A')}")

    # 加载数据
    print("\n" + "=" * 80)
    print("Loading market data...")
    print("=" * 80)

    df = load_data(args.data)

    # 计算因子得分（使用训练期结束日期的数据）
    target_date = deploy_range[0]  # 使用部署期开始日期之前的数据
    df_scored = calculate_factor_scores(df, expressions, weights, target_date=None)

    # 选择TopK股票
    print("\n" + "=" * 80)
    print("Selecting top stocks...")
    print("=" * 80)

    # 使用最后可用日期
    last_date = df_scored['date'].max()
    portfolio = select_top_stocks(df_scored, top_k=args.top_k, date=last_date)

    # 显示结果
    print("\n" + "=" * 80)
    print(f"Top {args.top_k} Stock Portfolio for {deploy_month}")
    print("=" * 80)
    print(portfolio.to_string(index=False))

    # 统计信息
    print("\n" + "=" * 80)
    print("Portfolio Statistics")
    print("=" * 80)
    print(f"Total stocks: {len(portfolio)}")
    print(f"Score range: [{portfolio['factor_score'].min():.4f}, {portfolio['factor_score'].max():.4f}]")
    print(f"Mean score: {portfolio['factor_score'].mean():.4f}")
    print(f"Median score: {portfolio['factor_score'].median():.4f}")

    # 保存
    if args.output is None:
        args.output = f"portfolio_{deploy_month}.csv"

    portfolio.to_csv(args.output, index=False)
    print(f"\nPortfolio saved to: {args.output}")


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime, timedelta
from scipy.stats import mstats
from sklearn.linear_model import LinearRegression

def load_real_lean_data(symbol, symbol_path, start_date, end_date):
    """
    Load real Lean data with correct timestamp parsing.
    Timestamps are milliseconds from midnight.
    """
    all_data = []
    current_date = start_date
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        zip_file = os.path.join(symbol_path, f"{date_str}_trade.zip")
        csv_file_in_zip = f"{date_str}_{symbol.lower()}_minute_trade.csv"
        
        if os.path.exists(zip_file):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    with zip_ref.open(csv_file_in_zip) as csv_file:
                        df = pd.read_csv(csv_file, header=None, 
                                       names=['time', 'open', 'high', 'low', 'close', 'volume'])
                        
                        date = pd.to_datetime(date_str, format='%Y%m%d')
                        df['datetime'] = date + pd.to_timedelta(df['time'], unit='ms')
                        
                        for col in ['open', 'high', 'low', 'close']:
                            df[col] = df[col] / 10000.0
                        
                        df['symbol'] = symbol
                        all_data.append(df)
            except Exception as e:
                # It's common for some dates or symbols to be missing, so we just note it.
                # print(f"Info: Could not load {symbol} data for {date_str}: {e}")
                pass
        
        current_date += timedelta(days=1)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def load_all_symbols_data(symbols, data_path, start_date, end_date):
    """
    Load data for all symbols and combine into a single DataFrame.
    """
    all_symbol_data = []
    
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        symbol_path = os.path.join(data_path, symbol.lower())
        
        if not os.path.exists(symbol_path):
            print(f"Path not found for {symbol}: {symbol_path}")
            continue
            
        df = load_real_lean_data(symbol, symbol_path, start_date, end_date)
        
        if not df.empty:
            all_symbol_data.append(df)
            print(f"  Loaded {len(df)} records for {symbol}")
    
    if all_symbol_data:
        combined_data = pd.concat(all_symbol_data, ignore_index=True)
        print(f"\nTotal records loaded: {len(combined_data)}")
        return combined_data
    else:
        print("No data loaded for any symbols")
        return pd.DataFrame()

def aggregate_to_am_pm(
    data_path: str,
    shares_file: str,
    start_date_str: str,
    end_date_str: str,
) -> pd.DataFrame:
    """
    Reads minute-level stock data, aggregates it into two sessions (AM and PM),
    and calculates vwap, mktcap, and turnover.
    """
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # --- 1. Load Minute Data ---
    print("--- Step 1: Loading Minute Data ---")
    symbols = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if not symbols:
        print(f"Error: No symbol directories found in '{data_path}'")
        return pd.DataFrame()
    print(f"Found {len(symbols)} symbols.")
    
    df = load_all_symbols_data(symbols, data_path, start_date, end_date)
    if df.empty:
        print("Stopping: No minute data was loaded.")
        return pd.DataFrame()

    # --- 2. Load and Merge Shares Outstanding Data ---
    print("\n--- Step 2: Loading and Merging Shares Outstanding ---")
    try:
        shares_df = pd.read_csv(shares_file)
        shares_df = shares_df[['symbol', 'shares_outstanding']].drop_duplicates()
    except FileNotFoundError:
        print(f"Error: Shares outstanding file not found at '{shares_file}'")
        return pd.DataFrame()
    
    df = pd.merge(df, shares_df, on='symbol', how='left')
    df.dropna(subset=['shares_outstanding'], inplace=True)
    if df.empty:
        print("Stopping: No data left after merging with shares outstanding. Check symbol match.")
        return pd.DataFrame()
    print("Shares outstanding data merged successfully.")

    # --- 3. Prepare Data and Define Sessions ---
    print("\n--- Step 3: Defining AM/PM Sessions ---")
    df['datetime'] = df['datetime'].dt.tz_localize("America/New_York", ambiguous='infer')
    df['date'] = df['datetime'].dt.date
    df['session'] = np.where(df['datetime'].dt.hour < 12, 'AM', 'PM')
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    print("AM/PM sessions defined (Split at 12:00 ET).")

    # --- 4. Aggregate Data ---
    print("\n--- Step 4: Aggregating Data into Sessions ---")
    def calculate_vwap(group):
        if group['volume'].sum() > 0:
            return np.average(group['close'], weights=group['volume'])
        return np.nan

    grouped = df.groupby(['symbol', 'date', 'session'])
    vwap = grouped.apply(calculate_vwap).rename('vwap').reset_index()

    session_df = grouped.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        shares_outstanding=('shares_outstanding', 'first')
    ).reset_index()

    session_df = pd.merge(session_df, vwap, on=['symbol', 'date', 'session'])
    print("Aggregation to sessions complete.")

    # --- 5. Calculate Financial Metrics ---
    print("\n--- Step 5: Calculating Financial Metrics ---")
    session_df['mktcap'] = session_df['close'] * session_df['shares_outstanding']
    session_df['turnover'] = session_df.apply(
        lambda row: row['volume'] / row['shares_outstanding'] if row['shares_outstanding'] > 0 else 0,
        axis=1
    )
    print("Calculated 'mktcap' and 'turnover'.")

    # --- 6. Finalize Data ---
    print("\n--- Step 6: Finalizing Data ---")
    session_df['timestamp'] = (
        pd.to_datetime(session_df['date']).dt.tz_localize("America/New_York")
        + pd.to_timedelta(session_df['session'].map({'AM': '12H', 'PM': '16H'}))
    )
    
    final_cols = [
        'symbol', 'timestamp', 'date', 'session', 'open', 'high', 'low', 'close',
        'volume', 'vwap', 'mktcap', 'turnover'
    ]
    session_df = session_df[final_cols]
    session_df = session_df.sort_values(by=['symbol', 'timestamp']).reset_index(drop=True)
    
    return session_df


import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# === 辅助：截面 winsorize（不把 NA 当 0） ===
def cs_winsorize(g: pd.Series, p: float = 0.01) -> pd.Series:
    x = g.copy()
    lo, hi = x.quantile(p), x.quantile(1 - p)
    return x.clip(lo, hi)

# === 辅助：滚动 OLS 残差（窗口尾点残差），索引与传入保持一致 ===
def rolling_ols_residuals_array(y: np.ndarray, X: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    resid = np.full(n, np.nan)
    for i in range(window, n + 1):
        yw = y[i - window:i]
        Xw = X[i - window:i]
        valid = (~np.isnan(yw)) & (~np.isnan(Xw).any(axis=1))
        if valid.sum() < max(5, Xw.shape[1] + 1):
            continue
        try:
            model = LinearRegression()
            model.fit(Xw[valid], yw[valid])
            # 用窗口最后一点做预测（与你原意一致）
            yhat = model.predict(X[i - 1:i])[0]
            resid[i - 1] = y[i - 1] - yhat
        except Exception:
            continue
    return resid

# === 核心：添加风格特征（AM/PM） ===
def add_style_features(
    session_df: pd.DataFrame,
    data_path: str,
    start_date_str: str,
    end_date_str: str
) -> pd.DataFrame:
    """
    Adds style-based features to the AM/PM session data (半日频).
    关键修正：
    - 半日收益用 close/open - 1
    - 基准ETF AM/PM 也用 open/close 聚合
    - IDIOVOL 组内滚动回归→残差→残差波动
    """
    print("\n--- Step 7: Adding Style Features ---")

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    annualization_factor = 504
    windows = [42, 126, 252]

    # --- 7.1 Load Benchmark Data ---
    print("--- Loading benchmark data ---")
    benchmark_symbols = ['VBR', 'QQQ', 'SPY', 'SPMO', 'SPHQ', 'SPYG']
    benchmark_df = load_all_symbols_data(benchmark_symbols, data_path, start_date, end_date)
    if benchmark_df.empty:
        print("Warning: No benchmark data loaded. Skipping feature creation.")
        return session_df

    # --- 7.2 Aggregate Benchmark Data (AM/PM open-close returns) ---
    print("--- Aggregating benchmark data ---")
    # 你的分钟数据若已是本地时区，这里仍显式设定，避免夏令时歧义
    benchmark_df['datetime'] = benchmark_df['datetime'].dt.tz_localize(
        "America/New_York", ambiguous='infer'
    )
    benchmark_df['date'] = benchmark_df['datetime'].dt.date
    benchmark_df['session'] = np.where(
        benchmark_df['datetime'].dt.hour < 12, 'AM', 'PM'
    )
    grouped_bench = benchmark_df.groupby(['symbol', 'date', 'session'])
    bench_agg = grouped_bench.agg(
        open=('open', 'first'),
        close=('close', 'last')
    ).reset_index()
    bench_agg = bench_agg.rename(columns={'symbol': 'benchmark'})
    bench_agg['benchmark_return'] = bench_agg['close'] / bench_agg['open'] - 1.0

    bench_pivot = bench_agg.pivot_table(
        index=['date', 'session'],
        columns='benchmark',
        values='benchmark_return'
    ).reset_index()
    bench_pivot.columns = [f"ret_{c}" if c not in ['date', 'session'] else c for c in bench_pivot.columns]
    benchmarks = [c.replace('ret_', '') for c in bench_pivot.columns if c.startswith('ret_')]

    # --- 7.3 Stock session returns (strict AM/PM close/open - 1) ---
    print("--- Preparing session returns for stocks ---")
    # 要求 session_df 含：symbol, date(或能转), session in {'AM','PM'}, open, close
    if 'date' not in session_df.columns:
        session_df['date'] = pd.to_datetime(session_df['datetime']).dt.date
    # 每条半日记录的收益
    session_df['ret_session'] = session_df['close'] / session_df['open'] - 1.0

    # 拆成 AM/PM 表并改列名
    df_am = (session_df[session_df['session'].eq('AM')]
             .copy()
             .rename(columns={'ret_session': 'return_am'}))
    df_pm = (session_df[session_df['session'].eq('PM')]
             .copy()
             .rename(columns={'ret_session': 'return_pm'}))

    # --- 7.4 Merge Benchmark Returns into AM/PM separately ---
    print("--- Merging benchmark returns ---")
    bench_am = bench_pivot[bench_pivot['session'].eq('AM')].drop(columns='session')
    bench_pm = bench_pivot[bench_pivot['session'].eq('PM')].drop(columns='session')

    df_am = df_am.merge(bench_am, on='date', how='left')
    df_pm = df_pm.merge(bench_pm, on='date', how='left')

    # --- 7.5 TE & CORR per benchmark/window ---
    print("--- Calculating TE/CORR per benchmark & window ---")
    df_am = df_am.sort_values(['symbol', 'date'])
    df_pm = df_pm.sort_values(['symbol', 'date'])

    for b in benchmarks:
        # 相对收益
        df_am[f'rel_ret_{b}'] = df_am['return_am'] - df_am[f'ret_{b}']
        df_pm[f'rel_ret_{b}'] = df_pm['return_pm'] - df_pm[f'ret_{b}']

        for w in windows:
            # TE
            df_am[f'TE_{b}_{w}_AM'] = (
                df_am.groupby('symbol')[f'rel_ret_{b}']
                .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor))
            )
            df_pm[f'TE_{b}_{w}_PM'] = (
                df_pm.groupby('symbol')[f'rel_ret_{b}']
                .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor))
            )

            # 同段相关（安全对齐）
            df_am[f'CORR_{b}_{w}_AM'] = (
                df_am.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_am'].rolling(w, min_periods=w//2).corr(g[f'ret_{b}']))
                .reset_index(level=0, drop=True)
            )
            df_pm[f'CORR_{b}_{w}_PM'] = (
                df_pm.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_pm'].rolling(w, min_periods=w//2).corr(g[f'ret_{b}']))
                .reset_index(level=0, drop=True)
            )

    # --- 7.6 合并 AM/PM 到同一行（按 symbol+date） ---
    print("--- Merging AM/PM panels into a single table ---")
    final_df = pd.merge(
        df_am, df_pm,
        on=['symbol', 'date'],
        suffixes=('_am', '_pm'),
        how='outer'
    ).sort_values(['symbol', 'date']).reset_index(drop=True)

    # --- 7.7 自身半日波动，用于 TEshare ---
    for w in windows:
        final_df[f'VOL_{w}_AM'] = (
            final_df.groupby('symbol')['return_am']
            .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor))
        )
        final_df[f'VOL_{w}_PM'] = (
            final_df.groupby('symbol')['return_pm']
            .transform(lambda s: s.rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor))
        )

    # --- 7.8 TEshare / DTE / RTE / CROSS_CORR / dTE ---
    print("--- Building TEshare/DTE/RTE/CROSS_CORR/dTE features ---")
    print("--- Building TEshare/DTE/RTE/CROSS_CORR/dTE features ---")
    for b in benchmarks:
        for w in windows:
            newcols = {}  # 先把这一批要添加的列放进 dict

            col_te_am = f'TE_{b}_{w}_AM'
            col_te_pm = f'TE_{b}_{w}_PM'

            # TEshare
            newcols[f'TEshare_{b}_{w}_AM'] = final_df[col_te_am] / (final_df[f'VOL_{w}_AM'] + 1e-8)
            newcols[f'TEshare_{b}_{w}_PM'] = final_df[col_te_pm] / (final_df[f'VOL_{w}_PM'] + 1e-8)

            # 截面 winsorize：AM、PM 分开做（注意不把 NA 当 0）
            tmp_am = pd.Series(newcols[f'TEshare_{b}_{w}_AM'], index=final_df.index)
            tmp_pm = pd.Series(newcols[f'TEshare_{b}_{w}_PM'], index=final_df.index)
            newcols[f'TEshare_{b}_{w}_AM'] = (
                tmp_am.groupby(final_df['date']).apply(cs_winsorize).reset_index(level=0, drop=True)
            )
            newcols[f'TEshare_{b}_{w}_PM'] = (
                tmp_pm.groupby(final_df['date']).apply(cs_winsorize).reset_index(level=0, drop=True)
            )

            # DTE / RTE
            newcols[f'DTE_{b}_{w}'] = final_df[col_te_am] - final_df[col_te_pm]
            newcols[f'RTE_{b}_{w}'] = final_df[col_te_am] / (final_df[col_te_pm] + 1e-8)

            # 跨段相关
            cross_am_pm = (
                final_df.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_am'].rolling(w).corr(g[f'ret_{b}_pm']))
                .reset_index(level=0, drop=True)
            )
            cross_pm_am = (
                final_df.groupby('symbol', group_keys=False)
                .apply(lambda g: g['return_pm'].rolling(w).corr(g[f'ret_{b}_am']))
                .reset_index(level=0, drop=True)
            )
            newcols[f'CROSS_CORR_{b}_{w}_AM_PM'] = cross_am_pm
            newcols[f'CROSS_CORR_{b}_{w}_PM_AM'] = cross_pm_am

            # dTE（10 个半日差分）
            newcols[f'dTE_{b}_{w}_AM'] = final_df.groupby('symbol')[col_te_am].transform(lambda s: s.diff(10))
            newcols[f'dTE_{b}_{w}_PM'] = final_df.groupby('symbol')[col_te_pm].transform(lambda s: s.diff(10))

            # —— 一次性追加这一小批列，避免碎片化 ——
            final_df = final_df.join(pd.DataFrame(newcols, index=final_df.index))

        # 每个基准做完，copy 一下进一步“去碎片”
        final_df = final_df.copy()


    print("--- Computing ARGMIN_TE labels ---")
    bench_arr = np.array(benchmarks, dtype=object)

    for w in windows:
        # AM 侧
        te_cols_am = [f'TE_{b}_{w}_AM' for b in benchmarks]
        vals_am = final_df[te_cols_am].to_numpy(dtype=float)
        has_val_am = ~np.isnan(vals_am)
        row_has_any_am = has_val_am.any(axis=1)

        safe_vals_am = np.where(np.isnan(vals_am), np.inf, vals_am)
        argmin_idx_am = safe_vals_am.argmin(axis=1)  # 不会再抛 All-NaN

        labels_am = np.full(len(final_df), np.nan, dtype=object)
        labels_am[row_has_any_am] = bench_arr[argmin_idx_am[row_has_any_am]]
        final_df[f'ARGMIN_TE_{w}_AM'] = labels_am

        # PM 侧
        te_cols_pm = [f'TE_{b}_{w}_PM' for b in benchmarks]
        vals_pm = final_df[te_cols_pm].to_numpy(dtype=float)
        row_has_any_pm = (~np.isnan(vals_pm)).any(axis=1)

        safe_vals_pm = np.where(np.isnan(vals_pm), np.inf, vals_pm)
        argmin_idx_pm = safe_vals_pm.argmin(axis=1)

        labels_pm = np.full(len(final_df), np.nan, dtype=object)
        labels_pm[row_has_any_pm] = bench_arr[argmin_idx_pm[row_has_any_pm]]
        final_df[f'ARGMIN_TE_{w}_PM'] = labels_pm

    # 去碎片
    final_df = final_df.copy()


    # --- 7.10 IDIOVOL：组内滚动回归→残差→残差波动（AM/PM 各自） ---
    print("--- Calculating IDIOVOL ---")
    X_cols_am = [f'ret_{b}_am' for b in benchmarks]
    X_cols_pm = [f'ret_{b}_pm' for b in benchmarks]

    for w in windows:
        print(f"  IDIOVOL window: {w}")

        # AM 侧：仅取有 AM 值的行
        mask_am = final_df['return_am'].notna()
        def _idiovol_am(g):
            y = g['return_am'].values
            X = g[X_cols_am].values
            resid = rolling_ols_residuals_array(y, X, w)
            resid_std = pd.Series(resid, index=g.index).rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor)
            return resid_std
        final_df.loc[mask_am, f'IDIOVOL_{w}_AM'] = (
            final_df[mask_am].sort_values(['symbol', 'date'])
            .groupby('symbol', group_keys=False)
            .apply(_idiovol_am)
        )

        # PM 侧：仅取有 PM 值的行
        mask_pm = final_df['return_pm'].notna()
        def _idiovol_pm(g):
            y = g['return_pm'].values
            X = g[X_cols_pm].values
            resid = rolling_ols_residuals_array(y, X, w)
            resid_std = pd.Series(resid, index=g.index).rolling(w, min_periods=w//2).std() * np.sqrt(annualization_factor)
            return resid_std
        final_df.loc[mask_pm, f'IDIOVOL_{w}_PM'] = (
            final_df[mask_pm].sort_values(['symbol', 'date'])
            .groupby('symbol', group_keys=False)
            .apply(_idiovol_pm)
        )

    # --- 7.11 清理 ---
    print("--- Final cleanup ---")
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    print("Style feature engineering complete.")
    return final_df


# ================================
# 下面是你主程序中用到的 I/O 框架（保留不变即可）
# ================================
if __name__ == '__main__':
    # Configuration
    LEAN_DATA_PATH = "e:/factor/lean_project/data/equity/usa/minute"
    SHARES_FILE = "shares.csv"
    BASE_DATA_FILE = "am_pm_base_data.csv"
    OUTPUT_FILE = "daily_am_pm_data_with_features.csv"
    START_DATE = "2022-01-01"  # YYYY-MM-DD
    END_DATE = "2025-11-10"    # YYYY-MM-DD

    # --- Load or Aggregate Data ---
    if os.path.exists(BASE_DATA_FILE):
        print(f"--- Loading base data from {BASE_DATA_FILE} ---")
        base_df = pd.read_csv(BASE_DATA_FILE)
        base_df['date'] = pd.to_datetime(base_df['date']).dt.date
    else:
        print(f"--- {BASE_DATA_FILE} not found, running full aggregation ---")
        base_df = aggregate_to_am_pm(
            data_path=LEAN_DATA_PATH,
            shares_file=SHARES_FILE,
            start_date_str=START_DATE,
            end_date_str=END_DATE,
        )
        if not base_df.empty:
            print(f"\n--- Saving aggregated data to {BASE_DATA_FILE} --- ")
            base_df.to_csv(BASE_DATA_FILE, index=False)

    if base_df.empty:
        print("Stopping: No base data available.")
    else:
        featured_df = add_style_features(
            base_df,
            data_path=LEAN_DATA_PATH,
            start_date_str=START_DATE,
            end_date_str=END_DATE
        )

        print(f"\n--- Saving Final Data to {OUTPUT_FILE} ---")
        featured_df.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')

        print("\n--- Processing Summary ---")
        print(f"Total rows in final dataset: {len(featured_df)}")
        print("Final DataFrame columns:", featured_df.columns.tolist())
        print("\nFirst 5 rows of the final dataset:")
        print(featured_df.head(5))
        print("\nLast 5 rows of the final dataset:")
        print(featured_df.tail(5))
        print(f"\nSuccessfully created '{OUTPUT_FILE}'.")
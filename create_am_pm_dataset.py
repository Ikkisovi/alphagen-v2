import pandas as pd
import numpy as np
import os
import zipfile
from datetime import datetime, timedelta

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
    output_path: str,
    start_date_str: str,
    end_date_str: str,
) -> None:
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
        return
    print(f"Found {len(symbols)} symbols.")
    
    df = load_all_symbols_data(symbols, data_path, start_date, end_date)
    if df.empty:
        print("Stopping: No minute data was loaded.")
        return

    # --- 2. Load and Merge Shares Outstanding Data ---
    print("\n--- Step 2: Loading and Merging Shares Outstanding ---")
    try:
        shares_df = pd.read_csv(shares_file)
        # Assuming columns are 'symbol' and 'shares_outstanding'
        shares_df = shares_df[['symbol', 'shares_outstanding']].drop_duplicates()
    except FileNotFoundError:
        print(f"Error: Shares outstanding file not found at '{shares_file}'")
        return
    
    df = pd.merge(df, shares_df, on='symbol', how='left')
    df.dropna(subset=['shares_outstanding'], inplace=True)
    if df.empty:
        print("Stopping: No data left after merging with shares outstanding. Check symbol match.")
        return
    print("Shares outstanding data merged successfully.")

    # --- 3. Prepare Data and Define Sessions ---
    print("\n--- Step 3: Defining AM/PM Sessions ---")
    # Localize naive datetime to New York time
    df['datetime'] = df['datetime'].dt.tz_localize("America/New_York", ambiguous='infer')
    df['date'] = df['datetime'].dt.date
    
    # AM session is before 12:00 ET, PM session is 12:00 ET or later
    df['session'] = np.where(df['datetime'].dt.hour < 12, 'AM', 'PM')
    
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    print("AM/PM sessions defined (Split at 12:00 ET).")

    # --- 4. Aggregate Data ---
    print("\n--- Step 4: Aggregating Data into Sessions ---")
    
    # Define a function to calculate VWAP for each group
    def calculate_vwap(group):
        if group['volume'].sum() > 0:
            return np.average(group['close'], weights=group['volume'])
        return np.nan

    grouped = df.groupby(['symbol', 'date', 'session'])
    
    # Calculate VWAP separately
    vwap = grouped.apply(calculate_vwap).rename('vwap').reset_index()

    # Aggregate OHLCV and shares
    session_df = grouped.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
        shares_outstanding=('shares_outstanding', 'first')
    ).reset_index()

    # Merge VWAP back
    session_df = pd.merge(session_df, vwap, on=['symbol', 'date', 'session'])
    print("Aggregation to sessions complete.")

    # --- 5. Calculate Financial Metrics ---
    print("\n--- Step 5: Calculating Financial Metrics ---")
    session_df['mktcap'] = session_df['close'] * session_df['shares_outstanding']
    
    # Avoid division by zero for turnover
    session_df['turnover'] = session_df.apply(
        lambda row: row['volume'] / row['shares_outstanding'] if row['shares_outstanding'] > 0 else 0,
        axis=1
    )
    print("Calculated 'mktcap' and 'turnover'.")

    # --- 6. Finalize and Save ---
    print("\n--- Step 6: Finalizing and Saving Data ---")
    # Create a representative timestamp for each session for sorting
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

    print(f"Aggregation complete. Saving to {output_path}...")
    session_df.to_csv(output_path, index=False, float_format='%.4f')

    print("\n--- Aggregation Summary ---")
    print(f"Total rows in new dataset: {len(session_df)}")
    print("New DataFrame columns:", session_df.columns.tolist())
    print("\nFirst 5 rows of the new AM/PM dataset:")
    print(session_df.head(5))
    print("\nLast 5 rows of the new AM/PM dataset:")
    print(session_df.tail(5))
    print(f"\nSuccessfully created '{output_path}'.")


if __name__ == '__main__':
    # Configuration
    LEAN_DATA_PATH = "e:/factor/lean_project/data/equity/usa/minute"
    SHARES_FILE = "shares.csv"
    OUTPUT_FILE = "daily_am_pm_data.csv"
    START_DATE = "2023-01-01"  # YYYY-MM-DD
    END_DATE = "2023-12-31"    # YYYY-MM-DD

    aggregate_to_am_pm(
        data_path=LEAN_DATA_PATH,
        shares_file=SHARES_FILE,
        output_path=OUTPUT_FILE,
        start_date_str=START_DATE,
        end_date_str=END_DATE,
    )
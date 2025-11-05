import os
import pandas as pd
import yfinance as yf
import time

TICKERS_ORIGINAL = [
    "MU", "TTMI", "CDE", "KGC", "COMM", "STRL", "DXPE", "WLDN", "SSRM", "LRN",
    "UNFI", "MFC", "EAT", "EZPW", "ARQT", "WFC", "ORGO", "PYPL", "ALL", "LC",
    "QTWO", "CLS", "CCL", "AGX", "POWL", "PPC", "SYF", "ATGE", "BRK.B", "SFM",
    "SKYW", "BLBD", "RCL", "OKTA", "TWLO", "APP", "TMUS", "UBER", "CAAP", "GBBK",
    "NBIS", "RKLB", "INCY", "NVDA", "GOOGL", "AMZN"
]

print("正在从 yfinance 获取流通股本 (sharesOutstanding) 数据...")
shares_data = {}

for ticker_orig in TICKERS_ORIGINAL:
    print(f"  正在获取 {ticker_orig}...")
    try:
        ticker_yf = yf.Ticker(ticker_orig)
        shares = ticker_yf.info.get('sharesOutstanding')
        
        alpaca_ticker_format = ticker_orig.replace('BRK.B', 'BRK/B')
        
        if shares:
            shares_data[alpaca_ticker_format] = shares
            print(f"    - {ticker_orig}: {shares}")
        else:
            shares_data[alpaca_ticker_format] = None
            print(f"    - 未找到 {ticker_orig} 的 'sharesOutstanding' 数据。")
    
    except Exception as e:
        alpaca_ticker_format = ticker_orig.replace('BRK.B', 'BRK/B')
        shares_data[alpaca_ticker_format] = None
        print(f"    - 获取 {ticker_orig} 数据时出错: {e}")
    
    time.sleep(0.1)

print("流通股本数据获取完毕。")

shares_df = pd.DataFrame.from_dict(shares_data, orient='index', columns=['shares_outstanding'])
shares_df.index.name = 'symbol'

output_filename = "shares.csv"
print(f"正在将流通股本数据保存到 {output_filename} ...")
shares_df.to_csv(output_filename)
print(f"数据已成功保存到 {output_filename}")

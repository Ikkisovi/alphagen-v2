import argparse
import json
import time
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yfinance as yf
try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    class YFRateLimitError(Exception):
        """Fallback rate limit error if yfinance does not expose the exception."""
        pass


# Mapping between final symbol naming (used downstream) and yfinance tickers.
SYMBOL_TO_YFINANCE: Dict[str, str] = {
    "MU": "MU",
    "TTMI": "TTMI",
    "CDE": "CDE",
    "KGC": "KGC",
    "COMM": "COMM",
    "STRL": "STRL",
    "DXPE": "DXPE",
    "WLDN": "WLDN",
    "SSRM": "SSRM",
    "LRN": "LRN",
    "UNFI": "UNFI",
    "MFC": "MFC",
    "EAT": "EAT",
    "EZPW": "EZPW",
    "ARQT": "ARQT",
    "WFC": "WFC",
    "ORGO": "ORGO",
    "PYPL": "PYPL",
    "ALL": "ALL",
    "LC": "LC",
    "QTWO": "QTWO",
    "CLS": "CLS",
    "CCL": "CCL",
    "AGX": "AGX",
    "POWL": "POWL",
    "PPC": "PPC",
    "SYF": "SYF",
    "ATGE": "ATGE",
    "BRK/B": "BRK-B",
    "SFM": "SFM",
    "SKYW": "SKYW",
    "BLBD": "BLBD",
    "RCL": "RCL",
    "OKTA": "OKTA",
    "TWLO": "TWLO",
    "APP": "APP",
    "TMUS": "TMUS",
    "UBER": "UBER",
    "CAAP": "CAAP",
    "GBBK": "GBBK",
    "NBIS": "NBIS",
    "RKLB": "RKLB",
    "INCY": "INCY",
    "NVDA": "NVDA",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
}

# Build reverse mapping for convenience (yfinance ticker -> final symbol).
YFINANCE_TO_SYMBOL: Dict[str, str] = {yf_symbol: symbol for symbol, yf_symbol in SYMBOL_TO_YFINANCE.items()}


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def make_date_windows(start: str, end: str, max_days: int, overlap_days: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = start_ts
    delta = timedelta(days=max_days)
    overlap = timedelta(days=overlap_days)

    while current < end_ts:
        window_end = min(current + delta, end_ts)
        windows.append((current, window_end))
        current = window_end - overlap
    return windows


def reshape_downloaded_frame(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        stacked = raw.stack(level=0).reset_index().rename(columns={"level_1": "yf_symbol"})
    else:
        stacked = raw.reset_index()
        ticker = tickers[0] if tickers else None
        stacked["yf_symbol"] = ticker

    timestamp_col = stacked.columns[0]
    stacked.rename(columns={timestamp_col: "timestamp"}, inplace=True)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    stacked.rename(columns=rename_map, inplace=True)

    required_cols = ["timestamp", "yf_symbol", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in stacked.columns:
            stacked[col] = pd.NA

    stacked = stacked[required_cols + [col for col in ["adj_close"] if col in stacked.columns]]
    stacked["symbol"] = stacked["yf_symbol"].map(YFINANCE_TO_SYMBOL)
    stacked = stacked.dropna(subset=["symbol"]).reset_index(drop=True)
    return stacked


def download_chunk_with_backoff(
    chunk: List[str],
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
    pause: float,
    max_retries: int,
    retry_wait: float,
) -> Optional[pd.DataFrame]:
    chunk_frames: List[pd.DataFrame] = []
    joined_tickers = " ".join(chunk)

    for window_start, window_end in windows:
        window_label = f"{window_start.date()} → {window_end.date()}"
        print(f"  [Window] {joined_tickers} | {window_label}")
        attempt = 0
        success = False
        while attempt <= max_retries:
            try:
                data = yf.download(
                    tickers=joined_tickers,
                    start=window_start.strftime("%Y-%m-%d"),
                    end=(window_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="60m",
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    prepost=False,
                    progress=False,
                    threads=True,
                )
                frame = reshape_downloaded_frame(data, chunk)
                print(
                    f"    [OK] {len(frame)} rows fetched "
                    f"({len(frame['symbol'].unique()) if not frame.empty else 0} symbols)"
                )
                chunk_frames.append(frame)
                success = True
                break
            except YFRateLimitError as exc:
                attempt += 1
                if attempt > max_retries:
                    print(f"    [Error] YF rate limit persisted for {joined_tickers} window {window_label}: {exc}")
                    return None
                wait = retry_wait * attempt
                print(f"    [Retry] Rate limited (attempt {attempt}/{max_retries}); sleeping {wait:.1f}s")
                time.sleep(wait)
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    print(f"    [Error] Failed to download {joined_tickers} window {window_label}: {exc}")
                    return None
                wait = retry_wait * attempt
                print(f"    [Warn] Error downloading {joined_tickers} window {window_label}: {exc}; retrying after {wait:.1f}s")
                time.sleep(wait)

        if not success:
            return None

        if pause > 0:
            time.sleep(pause)

    if not chunk_frames:
        return pd.DataFrame()

    return pd.concat(chunk_frames, ignore_index=True)


def download_hourly_from_yfinance(
    start: str,
    end: str,
    chunk_size: int,
    pause: float,
    max_days_per_request: int,
    max_retries: int,
    retry_wait: float,
) -> pd.DataFrame:
    yf_symbols = list(SYMBOL_TO_YFINANCE.values())
    windows = make_date_windows(start, end, max_days_per_request)

    pending: Deque[List[str]] = deque(chunk for chunk in chunked(yf_symbols, chunk_size))
    collected: List[pd.DataFrame] = []

    while pending:
        chunk = pending.popleft()
        print(f"[Chunk] Attempting tickers: {', '.join(chunk)}")
        chunk_df = download_chunk_with_backoff(chunk, windows, pause, max_retries, retry_wait)

        if chunk_df is None:
            if len(chunk) > 1:
                mid = len(chunk) // 2
                left = chunk[:mid]
                right = chunk[mid:]
                print(f"[Chunk] Splitting chunk due to failures → {left} | {right}")
                if left:
                    pending.appendleft(left)
                if right:
                    pending.appendleft(right)
            else:
                print(f"[Chunk] Giving up on ticker {chunk[0]} after repeated failures.")
        else:
            print(
                f"[Chunk Summary] {len(chunk_df)} rows across {len(chunk_df['symbol'].unique())} symbols "
                f"before de-duplication."
            )
            collected.append(chunk_df)

    if not collected:
        print("[Result] No chunks produced data.")
        return pd.DataFrame()

    merged = pd.concat(collected, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True).dt.tz_convert("America/New_York")
    merged["vwap"] = (merged["high"] + merged["low"] + merged["close"]) / 3.0
    merged = merged.drop(columns=["adj_close"], errors="ignore")
    merged = (
        merged.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    print(
        f"[Result] Combined hourly dataset: {len(merged)} rows, "
        f"{len(merged['symbol'].unique())} symbols, "
        f"date range {merged['timestamp'].min()} → {merged['timestamp'].max()}"
    )
    return merged


def load_local_alpaca_json(path: Path) -> pd.DataFrame:
    with path.open("r") as handle:
        payload = json.load(handle)

    rows: List[Dict[str, object]] = []
    bars = payload.get("bars", {})

    for symbol, entries in bars.items():
        final_symbol = symbol
        for bar in entries:
            timestamp = pd.Timestamp(bar["t"]).tz_convert("America/New_York")
            rows.append(
                {
                    "symbol": final_symbol,
                    "timestamp": timestamp,
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v"),
                    "vwap": bar.get("vw"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def _read_cached_shares(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "symbol" not in df.columns and df.index.name == "symbol":
        df = df.reset_index()
    shares_map: Dict[str, Optional[float]] = {}
    for _, row in df.iterrows():
        symbol = str(row["symbol"])
        value = row.get("shares_outstanding")
        if pd.isna(value):
            shares_map[symbol] = None
            continue
        try:
            shares_map[symbol] = float(value)
        except (TypeError, ValueError):
            shares_map[symbol] = None
    return shares_map


def fetch_missing_shares(missing_symbols: List[str]) -> Dict[str, float]:
    fetched: Dict[str, float] = {}
    for symbol in missing_symbols:
        yf_symbol = SYMBOL_TO_YFINANCE[symbol]
        shares = None
        try:
            ticker = yf.Ticker(yf_symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                shares = fast_info.get("shares_outstanding") or fast_info.get("shares")
            if not shares:
                info = ticker.info
                shares = info.get("sharesOutstanding")
        except Exception as exc:
            print(f"  Warning: failed to fetch shares for {symbol} ({yf_symbol}): {exc}")
        fetched[symbol] = shares
        time.sleep(0.2)
    return fetched


def load_or_fetch_shares(cache_path: Path) -> pd.DataFrame:
    cached = _read_cached_shares(cache_path)
    required_symbols = list(SYMBOL_TO_YFINANCE.keys())
    missing = [symbol for symbol in required_symbols if symbol not in cached or pd.isna(cached[symbol])]

    if missing:
        print(f"[Shares] Fetching sharesOutstanding for {len(missing)} symbols via yfinance...")
        fetched = fetch_missing_shares(missing)
        cached.update(fetched)
    else:
        print("[Shares] Using cached shares.csv values.")

    rows = [{"symbol": symbol, "shares_outstanding": cached.get(symbol)} for symbol in required_symbols]
    shares_df = pd.DataFrame(rows)
    shares_df.to_csv(cache_path, index=False)
    filled = shares_df["shares_outstanding"].notna().sum()
    print(f"[Shares] Saved to {cache_path} ({filled}/{len(shares_df)} populated).")
    return shares_df


def merge_price_and_shares(price_df: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
    merged = price_df.merge(shares_df, on="symbol", how="left")
    missing = merged["shares_outstanding"].isna().sum()
    if missing:
        print(f"[Merge] Warning: {missing} rows missing shares_outstanding after merge.")
    else:
        print("[Merge] All rows have shares_outstanding.")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hourly price dataset using yfinance and merge shares outstanding.")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD) for hourly download.")
    parser.add_argument("--end", default="2025-11-05", help="End date (YYYY-MM-DD) for hourly download.")
    parser.add_argument("--chunk-size", type=int, default=12, help="Number of tickers per yfinance request batch.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to pause between yfinance batches.")
    parser.add_argument("--max-days", type=int, default=350, help="Maximum calendar days per request to avoid yfinance limits.")
    parser.add_argument("--max-retries", type=int, default=4, help="Maximum retries per chunk/date window on failure.")
    parser.add_argument("--retry-wait", type=float, default=15.0, help="Base seconds to wait between retries (scaled by attempt).")
    parser.add_argument("--input-json", type=str, help="Optional Alpaca-style JSON payload to convert instead of downloading.")
    parser.add_argument("--output-bars", default="hourly_bars.csv", help="Path to save the raw hourly bars CSV.")
    parser.add_argument("--output-merged", default="hourly_stock_data_with_shares.csv", help="Path to save the merged dataset.")
    parser.add_argument("--shares-cache", default="shares.csv", help="Path for cached shares outstanding values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    if args.input_json:
        input_path = Path(args.input_json)
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {input_path}")
        print(f"Loading Alpaca-style data from {input_path}")
        price_df = load_local_alpaca_json(input_path)
    else:
        print(f"Downloading hourly data from yfinance ({args.start} → {args.end})")
        price_df = download_hourly_from_yfinance(
            args.start,
            args.end,
            args.chunk_size,
            args.pause,
            args.max_days,
            args.max_retries,
            args.retry_wait,
        )

    if price_df.empty:
        print("No price data available. Exiting.")
        return

    price_df.to_csv(args.output_bars, index=False)
    print(f"Saved hourly bars to {args.output_bars} ({len(price_df)} rows)")

    shares_cache = Path(args.shares_cache)
    shares_df = load_or_fetch_shares(shares_cache)

    merged_df = merge_price_and_shares(price_df, shares_df)
    merged_df.to_csv(args.output_merged, index=False)
    print(f"Merged dataset saved to {args.output_merged} ({len(merged_df)} rows)")

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()

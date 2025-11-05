#!/usr/bin/env python3
"""
Utility for converting the manually curated fundamental snapshot text file into
structured datasets that the Lean/AlphaGen pipeline can consume.

Workflow:
    1. Parse the raw `uniorganize_fundamental_data.txt` dump into a clean
       pandas DataFrame.
    2. Fetch (and forward fill) shares outstanding history for the universe via
       yfinance, falling back to static values when the full history is
       unavailable.
    3. Merge the datasets and export both a consolidated parquet file (for
       feature engineering/training) and per-symbol CSV files that can be read
       by a Lean custom data class at runtime.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


RAW_COLUMNS: List[str] = [
    "date",
    "symbol",
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_to_ebitda",
    "ev_to_revenue",
    "ev_to_fcf",
    "earnings_yield",
    "fcf_yield",
    "sales_yield",
    "forward_pe_ratio",
]

# yfinance does not recognise dots in tickers (e.g. BRK.B)
YF_SYMBOL_OVERRIDES: Dict[str, str] = {
    "BRK.B": "BRK-B",
}


def _clean_number(token: str) -> float:
    """Convert the string token from the txt dump into a float."""
    token = token.replace(",", "")
    if token.lower() == "nan":
        return np.nan
    try:
        return float(token)
    except ValueError as exc:  # pragma: no cover - defensive logging
        raise ValueError(f"Unable to parse numeric token '{token}'") from exc


def parse_fundamental_txt(path: Path) -> pd.DataFrame:
    """Parse the manually saved txt dump into a tidy DataFrame."""
    rows: List[List[float]] = []
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("(rows"):
                continue
            tokens = stripped.split()
            if not tokens[0].isdigit():
                continue
            if len(tokens) != len(RAW_COLUMNS) + 1:
                raise ValueError(f"Unexpected column count ({len(tokens)}) in line: {stripped}")
            row = [tokens[1], tokens[2]] + [_clean_number(tok) for tok in tokens[3:]]
            rows.append(row)

    df = pd.DataFrame(rows, columns=RAW_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["symbol", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_shares_outstanding(symbols: Iterable[str], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch shares outstanding history for each symbol and align to the provided dates.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "yfinance is required for fetching shares outstanding. "
            "Install it with `pip install yfinance`."
        ) from exc

    symbols = list(symbols)
    start, end = dates.min(), dates.max()
    aligned_dates = pd.DatetimeIndex(pd.to_datetime(dates).unique()).sort_values()
    records: List[pd.DataFrame] = []
    missing: List[str] = []

    print(f"Fetching shares outstanding for {len(symbols)} symbols "
          f"({start.date()} -> {end.date()})")

    for symbol in sorted(set(symbols)):
        yf_symbol = YF_SYMBOL_OVERRIDES.get(symbol, symbol)
        ticker = yf.Ticker(yf_symbol)
        series = None
        try:
            series = ticker.get_shares_full(start=start.strftime("%Y-%m-%d"),
                                            end=end.strftime("%Y-%m-%d"))
        except Exception as exc:  # pragma: no cover - network variability
            print(f"  [WARN] {symbol}: get_shares_full failed: {exc}")

        if series is not None and not series.empty:
            shares = (
                series.to_frame(name="shares_outstanding")
                .apply(pd.to_numeric, errors="coerce")
            )
            shares.index = pd.to_datetime(shares.index).tz_localize(None)
            shares = shares[~shares.index.duplicated(keep="last")]
            shares.index = shares.index.normalize()
            shares = shares.groupby(level=0).last()
        else:
            fallback_value = np.nan
            try:
                info = ticker.info
                fallback_value = float(info.get("sharesOutstanding") or np.nan)
            except Exception as exc:  # pragma: no cover - network variability
                print(f"  [WARN] {symbol}: unable to read fallback info: {exc}")
            shares = pd.DataFrame(
                {"shares_outstanding": [fallback_value]},
                index=pd.DatetimeIndex([end]),
            )

        shares = shares.reindex(aligned_dates).sort_index()
        shares["shares_outstanding"] = shares["shares_outstanding"].ffill().bfill()

        if shares["shares_outstanding"].isna().all():
            missing.append(symbol)

        symbol_df = shares.reset_index().rename(columns={"index": "date"})
        symbol_df["symbol"] = symbol
        records.append(symbol_df)

    combined = pd.concat(records, ignore_index=True)

    if missing:
        print(f"  [WARN] Missing shares_outstanding data for: {', '.join(missing)}")

    return combined


def export_custom_csv(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Write per-symbol CSV files that can be consumed by a Lean custom data type.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "time",
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "ev_to_ebitda",
        "ev_to_revenue",
        "ev_to_fcf",
        "earnings_yield",
        "fcf_yield",
        "sales_yield",
        "forward_pe_ratio",
        "shares_outstanding",
        "market_cap",
        "turnover",
    ]

    for symbol, symbol_df in df.groupby("symbol"):
        target_path = output_dir / f"{symbol}.csv"
        payload = symbol_df.sort_values("date").copy()
        payload["time"] = payload["date"].dt.strftime("%Y%m%d %H:%M")
        payload["market_cap"] = np.nan
        payload["turnover"] = np.nan
        payload.to_csv(target_path, index=False, columns=columns)
        print(f"  wrote {len(payload)} rows to {target_path}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build structured fundamental datasets")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("uniorganize_fundamental_data.txt"),
        help="Path to the raw txt dump",
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=Path("lean_project/data/fundamentals/fundamentals.parquet"),
        help="Combined parquet output for training/feature engineering",
    )
    parser.add_argument(
        "--custom-data-dir",
        type=Path,
        default=Path("lean_project/data/equity/usa/custom/manual_fundamentals"),
        help="Directory for per-symbol CSV exports for Lean custom data",
    )
    args = parser.parse_args(argv)

    df = parse_fundamental_txt(args.input)
    print(f"Parsed {len(df)} rows covering {df['symbol'].nunique()} symbols "
          f"from {df['date'].min().date()} to {df['date'].max().date()}")

    share_df = fetch_shares_outstanding(df["symbol"].unique(), df["date"])
    merged = (
        df.merge(share_df, on=["date", "symbol"], how="left")
        .sort_values(["symbol", "date"])
    )
    merged["shares_outstanding"] = merged.groupby("symbol")["shares_outstanding"].ffill().bfill()

    args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.parquet_output, index=False)
    print(f"Wrote consolidated parquet to {args.parquet_output}")

    export_custom_csv(merged, args.custom_data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

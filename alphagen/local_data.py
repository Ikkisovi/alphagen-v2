"""
Utilities for wrapping locally aggregated OHLCV data into AlphaGen-compatible
`StockData` objects without requiring a live Qlib installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch

from alphagen_qlib import stock_data as stock_data_mod
from alphagen_qlib.stock_data import FeatureType, StockData


@dataclass
class LocalDataConfig:
    """Configuration for converting the local dataset."""

    max_backtrack_days: int = 120
    max_future_days: int = 40
    features: Sequence[FeatureType] = (
        FeatureType.OPEN,
        FeatureType.CLOSE,
        FeatureType.HIGH,
        FeatureType.LOW,
        FeatureType.VOLUME,
        FeatureType.VWAP,
        FeatureType.PE_RATIO,
        FeatureType.PB_RATIO,
        FeatureType.PS_RATIO,
        FeatureType.EV_TO_EBITDA,
        FeatureType.EV_TO_REVENUE,
        FeatureType.EV_TO_FCF,
        FeatureType.EARNINGS_YIELD,
        FeatureType.FCF_YIELD,
        FeatureType.SALES_YIELD,
        FeatureType.FORWARD_PE_RATIO,
        FeatureType.SHARES_OUTSTANDING,
        FeatureType.MARKET_CAP,
        FeatureType.TURNOVER,
    )
    device: torch.device = torch.device("cpu")
    fundamental_path: Path | None = None


def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse intraday bars to daily OHLCV plus VWAP for each symbol.

    If the data is already daily (one row per date per symbol),
    skip aggregation and just ensure proper structure.
    """

    df = df.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    df["date"] = df["datetime"].dt.normalize()

    # Check if data is already daily resolution
    # Daily data has exactly one row per (date, symbol) pair
    grouped_counts = df.groupby(["date", "symbol"]).size()
    is_daily = (grouped_counts == 1).all()

    if is_daily:
        # Data is already daily, just ensure we have the required columns
        print("  Detected daily resolution data - skipping aggregation")
        daily = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()

        # Calculate VWAP if not present (for daily data, VWAP often equals close or is approximated)
        if "vwap" not in df.columns:
            # Use close as approximation for VWAP if not available
            daily["vwap"] = df["close"]
        else:
            daily["vwap"] = df["vwap"]

        daily["date"] = pd.to_datetime(daily["date"])
        return daily

    # Data is intraday - perform aggregation
    print("  Detected intraday data - aggregating to daily")
    df["pv"] = df["close"] * df["volume"]

    grouped = df.groupby(["date", "symbol"], sort=True)
    daily = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        pv=("pv", "sum"),
    )
    daily["vwap"] = daily["pv"] / daily["volume"].replace({0: np.nan})
    daily = daily.drop(columns=["pv"]).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _load_fundamental_table(path: Path) -> pd.DataFrame:
    """Load the preprocessed fundamental table from disk."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fundamental dataset not found: {path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        table = pd.read_parquet(path)
    else:
        table = pd.read_csv(path)

    table["date"] = pd.to_datetime(table["date"])
    table["symbol"] = table["symbol"].astype(str)
    return table


def _attach_fundamentals(
    daily: pd.DataFrame,
    cfg: LocalDataConfig,
    column_map: Mapping[FeatureType, str],
) -> pd.DataFrame:
    """
    Merge the fundamental dataset (if configured) into the aggregated OHLC data.
    """
    if cfg.fundamental_path is None:
        return daily

    fundamentals = _load_fundamental_table(cfg.fundamental_path)
    required_cols = {
        column_map[f]
        for f in cfg.features
        if f in column_map and column_map[f] not in {"open", "high", "low", "close", "volume", "vwap"}
    }
    if FeatureType.MARKET_CAP in cfg.features or FeatureType.TURNOVER in cfg.features:
        required_cols.add("shares_outstanding")

    if not required_cols:
        return daily

    merge_cols = ["date", "symbol"] + sorted(required_cols.intersection(fundamentals.columns))
    merged = daily.merge(
        fundamentals[merge_cols],
        on=["date", "symbol"],
        how="left",
    ).sort_values(["symbol", "date"])

    for col in merge_cols:
        if col in {"date", "symbol"}:
            continue
        merged[col] = merged.groupby("symbol")[col].ffill().bfill()

    if "shares_outstanding" in merged.columns:
        if "market_cap" in column_map.values():
            merged["market_cap"] = merged["close"] * merged["shares_outstanding"]
        if "turnover" in column_map.values():
            denom = merged["shares_outstanding"].replace({0: np.nan})
            merged["turnover"] = merged["volume"] / denom
    else:
        if "market_cap" in column_map.values() and "market_cap" not in merged.columns:
            merged["market_cap"] = np.nan
        if "turnover" in column_map.values() and "turnover" not in merged.columns:
            merged["turnover"] = np.nan

    return merged


def _build_extended_dates(
    dates: Sequence[pd.Timestamp],
    max_backtrack: int,
    max_future: int,
) -> pd.Index:
    """
    Construct a padded date index so the resulting StockData can support
    back-references and forward targets.
    """

    pad_before = pd.date_range(
        end=dates[0] - pd.Timedelta(days=1),
        periods=max_backtrack,
        freq="D",
    )
    pad_after = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=max_future,
        freq="D",
    )
    return pd.Index(list(pad_before) + list(dates) + list(pad_after))


def _pivot_features(
    daily: pd.DataFrame,
    features: Sequence[FeatureType],
    dates: Sequence[pd.Timestamp],
    symbols: Sequence[str],
) -> np.ndarray:
    """
    Convert the aggregated DataFrame into a dense (days, features, stocks) cube.
    """

    column_aliases: Mapping[FeatureType, Sequence[str]] = {
        FeatureType.OPEN: ("open",),
        FeatureType.CLOSE: ("close",),
        FeatureType.HIGH: ("high",),
        FeatureType.LOW: ("low",),
        FeatureType.VOLUME: ("volume",),
        FeatureType.VWAP: ("vwap",),
        FeatureType.PE_RATIO: ("pe_ratio",),
        FeatureType.PB_RATIO: ("pb_ratio",),
        FeatureType.PS_RATIO: ("ps_ratio",),
        FeatureType.EV_TO_EBITDA: ("ev_to_ebitda",),
        FeatureType.EV_TO_REVENUE: ("ev_to_revenue",),
        FeatureType.EV_TO_FCF: ("ev_to_fcf",),
        FeatureType.EARNINGS_YIELD: ("earnings_yield",),
        FeatureType.FCF_YIELD: ("fcf_yield",),
        FeatureType.SALES_YIELD: ("sales_yield",),
        FeatureType.FORWARD_PE_RATIO: ("forward_pe_ratio",),
        FeatureType.SHARES_OUTSTANDING: ("shares_outstanding",),
        FeatureType.MARKET_CAP: ("market_cap", "mktcap"),
        FeatureType.TURNOVER: ("turnover",),
    }
    tensors = []
    for feature in features:
        aliases = column_aliases.get(feature, ())
        column = next((name for name in aliases if name in daily.columns), None)
        if column is None:
            pivot = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
        else:
            pivot = (
                daily.pivot(index="date", columns="symbol", values=column)
                .reindex(index=dates, columns=symbols)
                .values.astype(np.float32)
            )
        tensors.append(pivot)
    stacked = np.stack(tensors, axis=1)  # (days, features, stocks)
    return stacked


def _dataframe_to_stock_data(
    frame: pd.DataFrame,
    cfg: LocalDataConfig,
) -> StockData:
    """Convert a normalized long-format frame into a ``StockData`` cube."""

    if frame.empty:
        raise ValueError("No rows available to build StockData from feature frame")

    normalized = frame.copy()
    normalized["symbol"] = normalized["symbol"].astype(str)
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized.sort_values(["symbol", "date"], inplace=True)

    symbols = sorted(normalized["symbol"].unique().tolist())
    dates = pd.Index(sorted(normalized["date"].unique()))

    cube = _pivot_features(normalized, cfg.features, dates, symbols)

    total_len = cfg.max_backtrack_days + cube.shape[0] + cfg.max_future_days
    padded = np.full(
        (total_len, cube.shape[1], cube.shape[2]),
        np.nan,
        dtype=np.float32,
    )
    start = cfg.max_backtrack_days
    padded[start : start + cube.shape[0]] = cube

    full_index = _build_extended_dates(dates, cfg.max_backtrack_days, cfg.max_future_days)
    tensor = torch.tensor(padded, dtype=torch.float32, device=cfg.device)

    stock = StockData(
        instrument="local",
        start_time=str(dates[0].date()),
        end_time=str(dates[-1].date()),
        max_backtrack_days=cfg.max_backtrack_days,
        max_future_days=cfg.max_future_days,
        features=list(cfg.features),
        device=cfg.device,
        preloaded_data=(tensor, full_index, pd.Index(symbols)),
    )
    return stock


_DEFAULT_SESSION_OFFSETS: MutableMapping[str, pd.Timedelta] = {
    "AM": pd.Timedelta(hours=12),
    "PM": pd.Timedelta(hours=16),
}


def _resolve_session_offsets(
    session_time_map: Mapping[str, object] | None,
) -> Mapping[str, pd.Timedelta]:
    """Normalize a mapping of session labels to ``pd.Timedelta`` offsets."""

    if session_time_map is None:
        return dict(_DEFAULT_SESSION_OFFSETS)

    resolved: dict[str, pd.Timedelta] = {}
    for key, value in session_time_map.items():
        label = str(key).upper()
        if isinstance(value, pd.Timedelta):
            offset = value
        elif isinstance(value, (int, float)):
            offset = pd.Timedelta(hours=float(value))
        else:
            offset = pd.to_timedelta(value)
        resolved[label] = offset
    return resolved


def load_local_stock_data(
    data_path: Path | str,
    *,
    config: LocalDataConfig | None = None,
) -> StockData:
    """
    Load local OHLCV pickle data into a `StockData` instance that AlphaGen can
    consume directly.
    """

    cfg = config or LocalDataConfig()
    stock_data_mod._QLIB_INITIALIZED = True  # type: ignore[attr-defined]

    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    daily = pd.read_csv(data_path)
    # The 'timestamp' column has the unique AM/PM timestamps we need for pivoting.
    # The existing 'date' column is just the day, so we drop it.
    daily.drop(columns=["date"], inplace=True)
    # We rename 'timestamp' to 'date' because the rest of the function expects that name.
    daily.rename(columns={"timestamp": "date"}, inplace=True)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    # Fundamental attachment and further aggregation is skipped as the CSV is pre-processed

    return _dataframe_to_stock_data(daily, cfg)


def build_feature_store_stock_data(
    frame: pd.DataFrame,
    *,
    config: LocalDataConfig | None = None,
    timezone: str | None = "UTC",
    session_time_map: Mapping[str, object] | None = None,
) -> StockData:
    """Convert a feature-store long table into ``StockData`` with AM/PM ordering."""

    cfg = config or LocalDataConfig()

    normalized = frame.copy()
    if "session" not in normalized.columns:
        raise ValueError("Feature store frame must contain a 'session' column")

    normalized["session"] = normalized["session"].astype(str).str.upper()
    offsets = _resolve_session_offsets(session_time_map)

    unknown_sessions = set(normalized["session"].unique()) - set(offsets.keys())
    if unknown_sessions:
        raise ValueError(
            "Encountered sessions without configured offsets: "
            + ", ".join(sorted(unknown_sessions))
        )

    base_dates = pd.to_datetime(normalized["date"])
    offsets_series = normalized["session"].map(offsets)
    if offsets_series.isna().any():
        raise ValueError("Session offset mapping produced NaN values")

    timestamps = base_dates + offsets_series
    timestamps = pd.to_datetime(timestamps)
    if timezone:
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize(timezone)
        else:
            timestamps = timestamps.dt.tz_convert(timezone)

    normalized["date"] = timestamps
    normalized.drop(columns=["session"], inplace=True)

    return _dataframe_to_stock_data(normalized, cfg)


__all__ = [
    "LocalDataConfig",
    "load_local_stock_data",
    "build_feature_store_stock_data",
]

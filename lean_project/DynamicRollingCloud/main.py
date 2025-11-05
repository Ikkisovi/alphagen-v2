# region imports
from AlgorithmImports import *
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from factor_engine import FactorEngine
# endregion


class DynamicRollingStrategy(QCAlgorithm):
    """Dynamic factor strategy that hot-loads monthly expressions from ObjectStore."""

    def Initialize(self):
        # Basic configuration
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2025, 10, 31)
        self.SetCash(100000)
        self.SetBenchmark("SPY")

        # Factor/ObjectStore state
        self.current_month: Optional[str] = None
        self.last_factor_lookup_failed: Optional[str] = None
        self.factor_expressions: List[str] = []
        self.factor_weights: np.ndarray = np.array([], dtype=float)
        self.factor_engine = FactorEngine(self)
        self.objectstore_prefix = "factors/"

        # Portfolio parameters
        self.lookback_days = 120
        self.max_history_days = 180
        self.top_quantile = 0.2
        self.max_position_size = 0.15
        self.max_position_count = 20
        self.min_dollar_volume = 1_000_000
        self.min_price = 5.0

        # Universe definition
        self.symbols = [
            'MU', 'TTMI', 'CDE', 'KGC', 'COMM', 'STRL', 'DXPE', 'WLDN', 'SSRM', 'LRN',
            'UNFI', 'MFC', 'EAT', 'EZPW', 'ARQT', 'WFC', 'ORGO', 'PYPL', 'ALL', 'LC',
            'QTWO', 'CLS', 'CCL', 'AGX', 'POWL', 'PPC', 'SYF', 'ATGE', 'BRK.B', 'SFM',
            'RKLB', 'SKYW', 'BLBD', 'RCL', 'OKTA', 'TWLO', 'APP', 'TMUS', 'UBER',
            'CAAP', 'GBBK', 'NBIS'
        ]
        self.equity_symbols: Dict[str, Symbol] = {}

        # Add equities and enable fundamental data
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Raw
        for ticker in self.symbols:
            equity = self.AddEquity(ticker, Resolution.Daily)
            equity.SetDataNormalizationMode(DataNormalizationMode.Raw)
            self.equity_symbols[ticker] = equity.Symbol

        # Cache for fundamental data (to avoid repeated lookups)
        self.fundamental_cache: Dict[Symbol, Dict] = {}

        # Scheduling: load factors at noon after the month rolls and rebalance monthly
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.At(12, 0),
            lambda: self.CheckAndUpdateFactors()
        )
        self.Schedule.On(
            self.DateRules.MonthStart("SPY"),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

        # Initial factor load
        self.CheckAndUpdateFactors(force=True)
        self.Log("Dynamic Rolling Strategy initialized")

    def CheckAndUpdateFactors(self, force: bool = False):
        """Load the latest factor definitions from ObjectStore when the month changes."""
        current_month = self.Time.strftime("%Y_%m")
        if not force and current_month == self.current_month:
            return

        factor_key = f"{self.objectstore_prefix}{current_month}.json"
        if not self.ObjectStore.ContainsKey(factor_key):
            if self.last_factor_lookup_failed != current_month:
                self.Log(f"Factor file '{factor_key}' not found; retaining factors for {self.current_month or 'previous month'}")
                self.last_factor_lookup_failed = current_month
            return

        try:
            factor_json = self.ObjectStore.Read(factor_key)
            factor_data = json.loads(factor_json)
        except Exception as err:
            self.Error(f"Error reading factor payload for {current_month}: {err}")
            return

        expressions = factor_data.get('expressions', [])
        weights = factor_data.get('weights', [])
        if not expressions or not weights or len(expressions) != len(weights):
            self.Log(f"Factor payload '{factor_key}' missing expressions/weights; skipping update")
            return

        self.factor_expressions = expressions
        self.factor_weights = np.asarray(weights, dtype=float)
        self.factor_engine.compile(self.factor_expressions)
        self.current_month = current_month
        self.last_factor_lookup_failed = None

        train_ic = factor_data.get('train_ic')
        deploy_ic = factor_data.get('deploy_ic')
        preview = ', '.join(self.factor_expressions[:3])
        self.Log(f"Loaded {len(self.factor_expressions)} factors for {current_month}: {preview}{'...' if len(self.factor_expressions) > 3 else ''}")
        if train_ic is not None:
            deploy_str = f"{deploy_ic:.4f}" if isinstance(deploy_ic, (int, float)) else deploy_ic
            self.Log(f"   Train IC: {train_ic:.4f}; Deploy IC: {deploy_str}")

    def Rebalance(self):
        """Monthly rebalance using the latest factor scores."""
        if not self.factor_expressions or self.factor_weights.size == 0:
            self.Log("No factor definitions available; skipping rebalance")
            return

        tradable_symbols = self._get_tradable_universe()
        if not tradable_symbols:
            self.Log("No tradable symbols passed liquidity/price filters; skipping rebalance")
            return

        scores = self._calculate_factor_scores(tradable_symbols)
        if not scores:
            self.Log("No valid factor scores produced; skipping rebalance")
            return

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        max_positions = min(self.max_position_count, max(1, int(len(ordered) * self.top_quantile)))
        targets = ordered[:max_positions]
        long_symbols = [symbol for symbol, _ in targets]
        if not long_symbols:
            self.Log("Factor selection returned empty basket; skipping rebalance")
            return

        target_weight = min(1.0 / len(long_symbols), self.max_position_size)
        top_preview = [(sym.Value, round(score, 4)) for sym, score in ordered[:5]]
        self.Log(f"Selected {len(long_symbols)} symbols | target weight={target_weight:.4f} | top5={top_preview}")

        for symbol in list(self.Portfolio.Keys):
            if symbol not in long_symbols and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        for symbol in long_symbols:
            self.SetHoldings(symbol, target_weight)

    def _get_tradable_universe(self) -> List[Symbol]:
        tradable = []
        for ticker, symbol in self.equity_symbols.items():
            if not self.Securities.ContainsKey(symbol):
                continue
            security = self.Securities[symbol]
            if security.Price < self.min_price:
                continue
            if security.Volume * security.Price < self.min_dollar_volume:
                continue
            tradable.append(symbol)
        return tradable

    def _calculate_factor_scores(self, symbols: List[Symbol]) -> Dict[Symbol, float]:
        raw_vectors: Dict[Symbol, np.ndarray] = {}
        factor_count = len(self.factor_expressions)
        for symbol in symbols:
            bundle = self._build_history_bundle(symbol)
            if not bundle:
                continue
            factor_values = self.factor_engine.evaluate(bundle)
            if not factor_values or len(factor_values) != factor_count:
                continue
            vector = np.asarray(factor_values, dtype=float)
            if not np.any(np.isfinite(vector)):
                continue
            raw_vectors[symbol] = vector

        if not raw_vectors:
            return {}

        symbol_list = list(raw_vectors.keys())
        matrix = np.vstack([raw_vectors[symbol] for symbol in symbol_list])
        normalized = self._zscore_columns(matrix)

        scores = {}
        for idx, symbol in enumerate(symbol_list):
            score = float(np.dot(self.factor_weights, normalized[idx]))
            if np.isfinite(score):
                scores[symbol] = score
        return scores

    def _build_history_bundle(self, symbol: Symbol) -> Optional[Dict[str, np.ndarray]]:
        try:
            price_history = self.History(symbol, self.max_history_days, Resolution.Daily)
        except Exception as err:
            self.Debug(f"Price history error for {symbol.Value}: {err}")
            return None

        if price_history is None or price_history.empty:
            return None

        if isinstance(price_history.index, pd.MultiIndex):
            price_history = price_history.xs(symbol, level=0)

        price_history = price_history.sort_index().iloc[-self.lookback_days:]
        if price_history.empty or 'close' not in price_history.columns:
            return None

        price_df = price_history[['open', 'high', 'low', 'close', 'volume']].astype(float)
        price_df['vwap'] = price_history['vwap'].astype(float) if 'vwap' in price_history.columns else price_df['close']

        valuation_df = self._get_valuation_history(symbol)
        if valuation_df.empty:
            valuation_df = pd.DataFrame(index=price_df.index)
        valuation_df = valuation_df.reindex(price_df.index, method='ffill')

        data = {
            'open': price_df['open'].values,
            'high': price_df['high'].values,
            'low': price_df['low'].values,
            'close': price_df['close'].values,
            'volume': price_df['volume'].values,
            'vwap': price_df['vwap'].values
        }

        required_fields = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_to_ebitda', 'ev_to_revenue', 'ev_to_fcf',
            'earnings_yield', 'fcf_yield', 'sales_yield', 'forward_pe_ratio', 'market_cap',
            'shares_outstanding', 'turnover'
        ]

        for field in required_fields:
            if field in valuation_df:
                data[field] = valuation_df[field].to_numpy(dtype=float)
            else:
                data[field] = np.full(len(price_df), np.nan)

        if not np.isfinite(data.get('shares_outstanding', np.array([]))).any():
            market_cap = data.get('market_cap')
            close = data['close']
            with np.errstate(divide='ignore', invalid='ignore'):
                shares = np.where((market_cap > 0) & (close > 0), market_cap / close, np.nan)
            data['shares_outstanding'] = shares

        with np.errstate(divide='ignore', invalid='ignore'):
            turnover = np.where(data['shares_outstanding'] > 0, data['volume'] / data['shares_outstanding'], np.nan)
        data['turnover'] = turnover

        return data

    def _get_valuation_history(self, symbol: Symbol) -> pd.DataFrame:
        """
        Get valuation ratios for a symbol.
        Note: QuantConnect Cloud has limited support for historical fundamental data.
        This method returns empty DataFrame for now - fundamental factors will use NaN values.
        TODO: Implement proper fundamental data fetching for cloud deployment.
        """
        # For now, return empty DataFrame
        # This allows price-based factors to work while fundamental factors return NaN
        return pd.DataFrame()

    @staticmethod
    def _normalize_column_name(name: str) -> str:
        return ''.join(ch.lower() for ch in str(name) if ch.isalnum())

    @staticmethod
    def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros_like(matrix)
        for col_idx in range(matrix.shape[1]):
            column = matrix[:, col_idx]
            mask = np.isfinite(column)
            if mask.sum() < 2:
                result[mask, col_idx] = 0.0
                continue
            mean = column[mask].mean()
            std = column[mask].std()
            if std < 1e-8:
                result[mask, col_idx] = 0.0
            else:
                result[mask, col_idx] = (column[mask] - mean) / std
            result[~mask, col_idx] = 0.0
        return result

    def OnData(self, data: Slice):
        """No per-tick processing required; scheduling handles rebalances."""
        return

    def OnEndOfAlgorithm(self):
        self.Log(f"\n{'='*60}")
        self.Log("Algorithm finished")
        self.Log(f"Final month: {self.current_month}")
        self.Log(f"Final portfolio value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"{'='*60}")

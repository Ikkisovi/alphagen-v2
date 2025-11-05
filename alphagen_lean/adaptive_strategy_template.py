"""
Adaptive AlphaGen Strategy for Lean Live Trading

This strategy:
1. Runs live trading with current factors
2. Every month, triggers AlphaGen retraining
3. Updates factors dynamically (with optional restart)
4. Continues trading with new factors

Usage:
    lean live deploy --environment "Paper Trading"
"""

from AlgorithmImports import *
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import importlib


class AdaptiveAlphaGenStrategy(QCAlgorithm):
    """
    自适应AlphaGen策略 - 定期重新训练因子
    """

    def Initialize(self):
        # 基本设置
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)
        self.SetBenchmark("SPY")

        # 路径配置
        self.alphagen_root = Path(r"E:\factor\alphagen")
        self.config_file = self.alphagen_root / "live_trading_config.json"
        self.factor_module_path = self.alphagen_root / "live_factors"
        self.factor_module_path.mkdir(exist_ok=True)

        # 加载配置
        self.load_config()

        # 添加股票
        self.symbols = self.config['symbols']
        self.equity_symbols = {}
        for ticker in self.symbols:
            try:
                symbol = self.AddEquity(ticker, Resolution.Minute).Symbol
                self.equity_symbols[ticker] = symbol
            except:
                self.Debug(f"Failed to add: {ticker}")

        self.Debug(f"Added {len(self.equity_symbols)} symbols")

        # 初始化数据聚合器
        sys.path.insert(0, str(self.alphagen_root))
        from alphagen_lean.templates.data_aggregator import MinuteDataAggregator
        self.data_aggregator = MinuteDataAggregator(
            list(self.equity_symbols.keys()),
            lookback_days=60
        )

        # 加载因子计算器（动态导入）
        self.load_factor_calculator()

        # 组合构造器
        from alphagen_lean.templates.portfolio_constructor import PortfolioConstructor
        self.portfolio_constructor = PortfolioConstructor(self)

        # 预热
        self.SetWarmUp(timedelta(days=50))

        # 调仓计划
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'MONTHLY')
        if self.rebalance_frequency == 'MONTHLY':
            self.Schedule.On(
                self.DateRules.MonthStart(),
                self.TimeRules.AfterMarketOpen(list(self.equity_symbols.values())[0], 30),
                self.Rebalance
            )

        # 重训练计划（每月末）
        self.Schedule.On(
            self.DateRules.MonthEnd(),
            self.TimeRules.At(16, 0),  # 收盘后
            self.TriggerRetraining
        )

        # 状态跟踪
        self.rebalance_count = 0
        self.retrain_count = 0
        self.last_config_update = datetime.now()
        self.first_rebalance_done = False

    def load_config(self):
        """加载配置文件"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            self.Debug(f"Loaded config: version={self.config.get('version', 'unknown')}")
        else:
            # 默认配置
            self.config = {
                'version': 'v0',
                'symbols': [
                    'MU', 'TTMI', 'CDE', 'KGC', 'COMM', 'STRL', 'DXPE', 'WLDN',
                    'SSRM', 'LRN', 'UNFI', 'MFC', 'EAT', 'EZPW', 'ARQT', 'WFC'
                ],
                'rebalance_frequency': 'MONTHLY',
                'auto_retrain': True,
                'retrain_window_months': 12,
                'retrain_steps': 5000,
                'last_retrain_date': None,
            }
            self.save_config()

    def save_config(self):
        """保存配置文件"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def load_factor_calculator(self):
        """动态加载因子计算器"""
        version = self.config.get('version', 'v0')

        # 尝试加载动态生成的因子模块
        factor_module_file = self.factor_module_path / f"factor_calculator_{version}.py"

        if factor_module_file.exists():
            self.Debug(f"Loading factor calculator: {version}")

            # 动态导入
            spec = importlib.util.spec_from_file_location(
                f"factor_calculator_{version}",
                factor_module_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            self.factor_calculator = module.FactorCalculator(
                self.data_aggregator,
                algorithm=self
            )
        else:
            # 使用默认因子计算器（从seekingal_worldq）
            self.Debug(f"Using default factor calculator")
            from alphagen_lean.templates.factor_calculator import FactorCalculator
            self.factor_calculator = FactorCalculator(
                self.data_aggregator,
                algorithm=self
            )

    def OnData(self, data):
        """每分钟数据更新"""
        # 更新数据聚合器
        for ticker, symbol in self.equity_symbols.items():
            if data.Bars.ContainsKey(symbol):
                self.data_aggregator.update(ticker, self.Time, data.Bars[symbol])

        # 检查配置是否更新（热更新支持）
        if self.config_file.exists():
            file_mod_time = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            if file_mod_time > self.last_config_update:
                self.Debug(f"Config file updated! Reloading...")
                self.load_config()
                self.load_factor_calculator()
                self.last_config_update = file_mod_time

                # 立即重新计算仓位
                self.Rebalance()

        # 预热结束后首次调仓
        if not self.IsWarmingUp and not self.first_rebalance_done:
            self.Debug("="*60)
            self.Debug(f"Warmup completed! First rebalance at {self.Time}")
            self.Debug("="*60)
            self.first_rebalance_done = True
            self.Rebalance()

    def Rebalance(self):
        """调仓"""
        if self.IsWarmingUp:
            return

        self.rebalance_count += 1
        self.Debug(f"\n{'='*60}")
        self.Debug(f"Rebalance #{self.rebalance_count} @ {self.Time}")
        self.Debug(f"Factor version: {self.config.get('version', 'unknown')}")

        # 计算因子
        factors = self.factor_calculator.calculate_all_factors(
            list(self.equity_symbols.keys())
        )

        valid = {k: v for k, v in factors.items() if not np.isnan(v)}
        self.Debug(f"Valid factors: {len(valid)}/{len(factors)}")

        if len(valid) < 10:
            self.Debug("Not enough valid factors, skipping rebalance")
            return

        # 构建组合
        securities = {t: self.Securities[s] for t, s in self.equity_symbols.items()}
        weights = self.portfolio_constructor.construct_portfolio(factors, securities)

        self.Debug(f"Target positions: {len(weights)}")

        # 执行交易
        for ticker, symbol in self.equity_symbols.items():
            if ticker not in weights and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)

        for ticker, weight in weights.items():
            self.SetHoldings(self.equity_symbols[ticker], weight)

        self.Debug("="*60 + "\n")

    def TriggerRetraining(self):
        """触发AlphaGen重训练"""
        if not self.config.get('auto_retrain', True):
            self.Debug("Auto-retrain disabled in config")
            return

        self.retrain_count += 1
        self.Debug(f"\n{'='*60}")
        self.Debug(f"TRIGGERING ALPHAGEN RETRAINING #{self.retrain_count}")
        self.Debug(f"Current time: {self.Time}")
        self.Debug(f"{'='*60}\n")

        try:
            # 调用外部训练脚本
            train_script = self.alphagen_root / "scripts" / "retrain_for_live.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "--end-date", self.Time.strftime("%Y-%m-%d"),
                    "--window-months", str(self.config.get('retrain_window_months', 12)),
                    "--steps", str(self.config.get('retrain_steps', 5000)),
                    "--output-dir", str(self.factor_module_path),
                ],
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )

            if result.returncode == 0:
                self.Debug("✓ Retraining completed successfully!")
                self.Debug(f"Output: {result.stdout[-200:]}")  # 最后200字符

                # 更新配置
                self.config['version'] = f"v{self.retrain_count}"
                self.config['last_retrain_date'] = self.Time.strftime("%Y-%m-%d")
                self.save_config()

                # 重新加载因子（热更新）
                self.load_factor_calculator()

                self.Debug(f"Updated to factor version: {self.config['version']}")
            else:
                self.Debug(f"✗ Retraining failed!")
                self.Debug(f"Error: {result.stderr[-200:]}")

        except subprocess.TimeoutExpired:
            self.Debug("✗ Retraining timeout (>1 hour)")
        except Exception as e:
            self.Debug(f"✗ Retraining error: {e}")

    def OnEndOfAlgorithm(self):
        """结束时总结"""
        ret = (self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100
        self.Debug(f"\n{'='*60}")
        self.Debug(f"LIVE TRADING SUMMARY")
        self.Debug(f"{'='*60}")
        self.Debug(f"Total Return: {ret:.2f}%")
        self.Debug(f"Rebalances: {self.rebalance_count}")
        self.Debug(f"Retrainings: {self.retrain_count}")
        self.Debug(f"Final Factor Version: {self.config.get('version')}")
        self.Debug(f"Final Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"{'='*60}")

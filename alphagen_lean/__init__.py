"""
AlphaGen-Lean Integration Module

This module provides tools for rolling window training of AlphaGen factors
and automatic deployment to Lean backtesting framework.
"""

__version__ = "0.1.0"

from .data_prep import LeanDataLoader
from .window_manager import WindowManager
from .rolling_trainer import RollingTrainer
from .expression_converter import ExpressionConverter
from .lean_exporter import LeanExporter

__all__ = [
    "LeanDataLoader",
    "WindowManager",
    "RollingTrainer",
    "ExpressionConverter",
    "LeanExporter",
]

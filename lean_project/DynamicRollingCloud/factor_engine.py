# region imports
from AlgorithmImports import *
# endregion
"""
Expression converter: Convert AlphaGen expression strings to Python code.

This module parses AlphaGen expression strings (tree-structured mathematical expressions)
and generates executable Python code using NumPy operations.

Example:
    Input:  "Mean($close, 20d)"
    Output: "np.mean(h['close'][-20:])"
"""

import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


class ExpressionConverter:
    """
    Convert AlphaGen expression strings to Python function code.

    Supports:
    - Features: $close, $open, $high, $low, $volume, $vwap
    - Constants: 10.0, -0.5, etc.
    - Binary operators: Add, Sub, Mul, Div, Greater, Less, Pow
    - Unary operators: Abs, Log, Sign, Sqrt
    - Rolling operators: Mean, Sum, Std, Var, Max, Min, Med, Delta, Ref
    - Advanced operators: Mad, Corr, Cov, Rank, WMA, EMA
    """

    # Feature mapping
    FEATURES = {
        '$close': "h['close']",
        '$open': "h['open']",
        '$high': "h['high']",
        '$low': "h['low']",
        '$volume': "h['volume']",
        '$vwap': "h['vwap']",
        '$pe_ratio': "h['pe_ratio']",
        '$pb_ratio': "h['pb_ratio']",
        '$ps_ratio': "h['ps_ratio']",
        '$ev_to_ebitda': "h['ev_to_ebitda']",
        '$ev_to_revenue': "h['ev_to_revenue']",
        '$ev_to_fcf': "h['ev_to_fcf']",
        '$earnings_yield': "h['earnings_yield']",
        '$fcf_yield': "h['fcf_yield']",
        '$sales_yield': "h['sales_yield']",
        '$forward_pe_ratio': "h['forward_pe_ratio']",
        '$shares_outstanding': "h['shares_outstanding']",
        '$market_cap': "h['market_cap']",
        '$turnover': "h['turnover']",
    }

    def __init__(self):
        self.current_pos = 0
        self.expression = ""

    def convert(self, expression_str: str, function_name: str = "_factor") -> str:
        """
        Convert an AlphaGen expression to a Python function.

        Args:
            expression_str: AlphaGen expression string (e.g., "Mean($close, 20d)")
            function_name: Name of the generated function

        Returns:
            Python function code as string
        """
        # Reset state
        self.current_pos = 0
        self.expression = expression_str

        # Parse and generate code
        try:
            code = self._parse_expression()

            # Generate function (no leading indentation for exec())
            function_code = f'''def {function_name}(self, h):
    """{expression_str}"""
    try:
        return {code}
    except Exception as e:
        algo = getattr(self, 'algorithm', None)
        if algo is not None and hasattr(algo, 'Debug'):
            algo.Debug(f"Error in {function_name}: {{e}}")
        return np.nan'''

            return function_code

        except Exception as e:
            raise ValueError(f"Failed to convert expression '{expression_str}': {e}")

    def _parse_expression(self) -> str:
        """Parse a single expression (recursive)."""
        self._skip_whitespace()

        # Check if it's a function call (operator)
        if self._peek_char().isalpha():
            return self._parse_function()

        # Check if it's a feature ($xxx)
        elif self._peek_char() == '$':
            return self._parse_feature()

        # Check if it's a number (constant)
        elif self._peek_char().isdigit() or self._peek_char() == '-' or self._peek_char() == '.':
            return self._parse_number()

        # Check if it's a parenthesized expression
        elif self._peek_char() == '(':
            self._consume_char('(')
            expr = self._parse_expression()
            self._consume_char(')')
            return f"({expr})"

        else:
            raise ValueError(f"Unexpected character: {self._peek_char()} at position {self.current_pos}")

    def _parse_function(self) -> str:
        """Parse a function call (operator)."""
        # Read function name
        func_name = self._read_identifier()

        self._skip_whitespace()
        self._consume_char('(')

        # Parse arguments
        args = []
        while True:
            self._skip_whitespace()
            if self._peek_char() == ')':
                break

            arg = self._parse_argument()
            args.append(arg)

            self._skip_whitespace()
            if self._peek_char() == ',':
                self._consume_char(',')
            elif self._peek_char() == ')':
                break
            else:
                raise ValueError(f"Expected ',' or ')' at position {self.current_pos}")

        self._consume_char(')')

        # Generate code based on function name
        return self._generate_operator_code(func_name, args)

    def _parse_feature(self) -> str:
        """Parse a feature ($xxx)."""
        feature = self._read_identifier()
        if feature not in self.FEATURES:
            raise ValueError(f"Unknown feature: {feature}")

        # Most features return the last value
        # But some operators use the full array
        return self.FEATURES[feature]

    def _parse_number(self) -> str:
        """Parse a number (constant)."""
        start = self.current_pos
        if self._peek_char() == '-':
            self._advance()

        while self._peek_char().isdigit() or self._peek_char() == '.':
            self._advance()

        return self.expression[start:self.current_pos]

    def _parse_argument(self) -> Tuple[str, str]:
        """
        Parse an argument (can be expression or time delta).

        Returns:
            Tuple of (type, value) where type is 'expr' or 'time'
        """
        self._skip_whitespace()

        # Check if it's a time delta (e.g., "20d")
        # Look ahead to see if it matches pattern: digits followed by 'd'
        if self._is_time_delta():
            time_val = self._read_time_delta()
            return ('time', time_val)
        else:
            expr = self._parse_expression()
            return ('expr', expr)

    def _is_time_delta(self) -> bool:
        """Check if current position starts with a time delta (e.g., '20d')."""
        saved_pos = self.current_pos

        # Check if starts with digits
        if not self._peek_char().isdigit():
            return False

        # Advance through digits
        while self._peek_char().isdigit():
            self._advance()

        # Check if followed by 'd'
        is_time = self._peek_char() == 'd'

        # Restore position
        self.current_pos = saved_pos

        return is_time

    def _read_identifier(self) -> str:
        """Read an identifier (function name or feature)."""
        start = self.current_pos

        # First character can be $ or letter
        if self._peek_char() == '$':
            self._advance()

        while self._peek_char().isalnum() or self._peek_char() == '_':
            self._advance()

        return self.expression[start:self.current_pos]

    def _read_time_delta(self) -> str:
        """Read a time delta (e.g., '20d')."""
        start = self.current_pos

        while self._peek_char().isdigit():
            self._advance()

        if self._peek_char() == 'd':
            self._advance()
        else:
            raise ValueError(f"Expected 'd' at position {self.current_pos}")

        time_str = self.expression[start:self.current_pos]
        return time_str[:-1]  # Remove the 'd'

    def _generate_operator_code(self, func_name: str, args: List[Tuple[str, str]]) -> str:
        """
        Generate Python code for an operator.

        Args:
            func_name: Operator name (e.g., 'Mean', 'Add', 'Abs')
            args: List of (type, value) tuples

        Returns:
            Python code string
        """
        # Binary operators
        if func_name == 'Add':
            return f"({args[0][1]} + {args[1][1]})"
        elif func_name == 'Sub':
            return f"({args[0][1]} - {args[1][1]})"
        elif func_name == 'Mul':
            return f"({args[0][1]} * {args[1][1]})"
        elif func_name == 'Div':
            return f"({args[0][1]} / ({args[1][1]} + 1e-8))"
        elif func_name == 'Pow':
            return f"np.power({args[0][1]}, {args[1][1]})"
        elif func_name == 'Greater':
            return f"np.maximum({args[0][1]}, {args[1][1]})"
        elif func_name == 'Less':
            return f"np.minimum({args[0][1]}, {args[1][1]})"

        # Unary operators
        elif func_name == 'Abs':
            return f"np.abs({args[0][1]})"
        elif func_name == 'Log':
            return f"np.log(np.maximum({args[0][1]}, 1e-8))"
        elif func_name == 'Sign':
            return f"np.sign({args[0][1]})"
        elif func_name == 'Sqrt':
            return f"np.sqrt(np.maximum({args[0][1]}, 0))"

        # Rolling operators with time window
        elif func_name == 'Mean':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_mean({expr}, {window})"

        elif func_name == 'Sum':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_sum({expr}, {window})"

        elif func_name == 'Std':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_std({expr}, {window})"

        elif func_name == 'Var':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_var({expr}, {window})"

        elif func_name == 'Max':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_max({expr}, {window})"

        elif func_name == 'Min':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_min({expr}, {window})"

        elif func_name == 'Med':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_median({expr}, {window})"

        elif func_name == 'Delta':
            expr, window = self._extract_expr_and_window(args)
            return f"self._delta({expr}, {window})"

        elif func_name == 'Ref':
            expr, window = self._extract_expr_and_window(args)
            return f"self._ref({expr}, {window})"

        # Special operators
        elif func_name == 'Mad':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_mad({expr}, {window})"

        elif func_name == 'Rank':
            # Rank typically works on the latest value across stocks, not time-series
            # For single stock, we can use the time-series rank
            expr = args[0][1]
            return f"(pd.Series({expr}).rank(pct=True).iloc[-1] if len({expr}) > 0 else 0.5)"

        elif func_name == 'WMA':
            expr, window = self._extract_expr_and_window(args)
            return f"self._wma({expr}, {window})"

        elif func_name == 'EMA':
            expr, window = self._extract_expr_and_window(args)
            return f"self._ema({expr}, {window})"

        elif func_name == 'Corr':
            expr1, window1 = self._extract_expr_and_window([args[0], args[2]])
            expr2 = f"self._ensure_array({args[1][1]})"
            return (
                f"np.corrcoef({expr1}[-{window1}:], {expr2}[-{window1}:])[0, 1]"
                f" if len({expr1}) >= {window1} and len({expr2}) >= {window1} else 0"
            )

        elif func_name == 'Cov':
            expr1, window1 = self._extract_expr_and_window([args[0], args[2]])
            expr2 = f"self._ensure_array({args[1][1]})"
            return (
                f"np.cov({expr1}[-{window1}:], {expr2}[-{window1}:])[0, 1]"
                f" if len({expr1}) >= {window1} and len({expr2}) >= {window1} else 0"
            )

        else:
            raise ValueError(f"Unsupported operator: {func_name}")

    def _extract_expr_and_window(self, args: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Extract expression and window from arguments.

        Most rolling operators have signature: Op(expr, window)
        Returns: (expr_code, window_value)
        """
        if len(args) < 2:
            raise ValueError(f"Expected at least 2 arguments, got {len(args)}")

        expr_type, expr_code = args[0]
        window_type, window_value = args[1]

        if window_type != 'time':
            # If second arg is not a time delta, it might be a constant expression
            # Try to evaluate it as a number
            try:
                window_value = str(int(float(window_value)))
            except:
                raise ValueError(f"Expected time delta or numeric constant for window, got {window_value}")

        return f"self._ensure_array({expr_code})", window_value

    # Helper methods for parsing
    def _peek_char(self) -> str:
        """Peek at current character without consuming it."""
        if self.current_pos >= len(self.expression):
            return ''
        return self.expression[self.current_pos]

    def _peek_ahead(self, n: int = 10) -> str:
        """Peek ahead n characters."""
        return self.expression[self.current_pos:self.current_pos + n]

    def _advance(self):
        """Move to next character."""
        self.current_pos += 1

    def _consume_char(self, expected: str):
        """Consume expected character or raise error."""
        if self._peek_char() != expected:
            raise ValueError(f"Expected '{expected}' at position {self.current_pos}, got '{self._peek_char()}'")
        self._advance()

    def _skip_whitespace(self):
        """Skip whitespace characters."""
        while self._peek_char() in ' \t\n\r':
            self._advance()

    @staticmethod
    def generate_helper_functions() -> str:
        """
        Generate helper functions used by converted expressions.

        Returns:
            Python code for helper functions
        """
        return '''def _ensure_array(self, data):
    arr = np.asarray(data)
    if arr.ndim == 0:
        try:
            return np.array([float(arr)])
        except Exception:
            return np.array([0.0])
    try:
        return arr.astype(float, copy=False).reshape(-1)
    except Exception:
        return np.asarray(arr, dtype=float).reshape(-1)

def _sanitize_window(self, window, size):
    try:
        w = int(window)
    except Exception:
        w = int(float(window))
    if size <= 0:
        return 1
    return max(1, min(w, size))

def _rolling_mean(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    return float(np.mean(arr[-window:]))

def _rolling_sum(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return 0.0
    window = self._sanitize_window(window, arr.size)
    return float(np.sum(arr[-window:]))

def _rolling_std(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    return float(np.std(subset)) if subset.size > 0 else np.nan

def _rolling_var(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    return float(np.var(subset)) if subset.size > 0 else np.nan

def _rolling_max(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    return float(np.max(subset)) if subset.size > 0 else np.nan

def _rolling_min(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    return float(np.min(subset)) if subset.size > 0 else np.nan

def _rolling_median(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    return float(np.median(subset)) if subset.size > 0 else np.nan

def _rolling_mad(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    if subset.size == 0:
        return np.nan
    median = np.median(subset)
    return float(np.median(np.abs(subset - median)))

def _wma(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    weights = np.arange(1, subset.size + 1)
    return float(np.average(subset, weights=weights))

def _ema(self, data, window):
    arr = self._ensure_array(data)
    if arr.size == 0:
        return np.nan
    window = self._sanitize_window(window, arr.size)
    subset = arr[-window:]
    alpha = 2 / (window + 1)
    ema = subset[0]
    for val in subset[1:]:
        ema = alpha * val + (1 - alpha) * ema
    return float(ema)

def _ref(self, data, window):
    arr = self._ensure_array(data)
    try:
        w = int(window)
    except Exception:
        w = int(float(window))
    if arr.size < w or w <= 0:
        return np.nan
    return float(arr[-w])

def _delta(self, data, window):
    arr = self._ensure_array(data)
    try:
        w = int(window)
    except Exception:
        w = int(float(window))
    if arr.size < w or w <= 0:
        return 0.0
    return float(arr[-1] - arr[-w])'''


def convert_expression_list(expressions: List[str]) -> Tuple[List[str], str]:
    """
    Convert a list of AlphaGen expressions to Python function code.

    Args:
        expressions: List of AlphaGen expression strings

    Returns:
        Tuple of (function_codes, helper_code)
        - function_codes: List of Python function code strings
        - helper_code: Python code for helper functions
    """
    converter = ExpressionConverter()
    function_codes = []

    for i, expr in enumerate(expressions):
        func_name = f"_f{i+1}"
        try:
            func_code = converter.convert(expr, func_name)
            function_codes.append(func_code)
        except Exception as e:
            print(f"Warning: Failed to convert expression {i+1}: {expr}")
            print(f"  Error: {e}")
            error_msg = str(e).replace('"', '\\"')
            expr_escaped = expr.replace('"', '\\"')
            # Generate a fallback function that returns NaN
            fallback_code = (
                f'def {func_name}(self, h):\n'
                f'    """{expr_escaped}"""\n'
                f'    algo = getattr(self, "algorithm", None)\n'
                f'    if algo is not None and hasattr(algo, "Debug"):\n'
                f'        algo.Debug("Conversion failed for expression: {expr_escaped} ({error_msg})")\n'
                f'    return np.nan'
            )
            function_codes.append(fallback_code)

    helper_code = ExpressionConverter.generate_helper_functions()

    return function_codes, helper_code


class FactorEngine:
    """Compile and evaluate AlphaGen factor expressions inside Lean."""

    def __init__(self, algorithm) -> None:
        self.algorithm = algorithm
        self.expressions: List[str] = []
        self._compiled = None

    def compile(self, expressions: List[str]) -> None:
        self.expressions = expressions or []
        if not self.expressions:
            self._compiled = None
            return

        function_codes, helper_code = convert_expression_list(self.expressions)

        class_lines: List[str] = [
            "import numpy as np",
            "import pandas as pd",
            "class _CompiledFactorLibrary:",
            "    def __init__(self, algorithm):",
            "        self.algorithm = algorithm",
        ]

        for code in function_codes:
            class_lines.extend(("    " + line) if line else "" for line in code.splitlines())

        class_lines.extend(("    " + line) if line else "" for line in helper_code.splitlines())

        func_list = ", ".join(f"self._f{i+1}" for i in range(len(function_codes)))
        class_lines.extend([
            "    def evaluate(self, h):",
            "        results = []",
            f"        funcs = [{func_list}]" if func_list else "        funcs = []",
            "        for func in funcs:",
            "            try:",
            "                value = func(h)",
            "                if isinstance(value, (list, tuple, np.ndarray)):",
            "                    value = value[-1] if len(value) > 0 else np.nan",
            "                results.append(float(value))",
            "            except Exception as err:",
            "                algo = getattr(self, 'algorithm', None)",
            "                if algo is not None and hasattr(algo, 'Debug'):",
            '                    algo.Debug(f"Factor evaluation error: {err}")',
            "                results.append(np.nan)",
            "        return results",
        ])

        namespace: Dict[str, object] = {}
        exec("\n".join(class_lines), namespace)
        library_cls = namespace["_CompiledFactorLibrary"]
        self._compiled = library_cls(self.algorithm)

    def evaluate(self, history: Dict[str, np.ndarray]) -> List[float]:
        if not self._compiled:
            return []
        return self._compiled.evaluate(history)

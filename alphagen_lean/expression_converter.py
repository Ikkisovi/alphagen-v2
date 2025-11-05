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

            # Generate function
            function_code = f'''    def {function_name}(self, h):
        """{expression_str}"""
        try:
            return {code}
        except Exception as e:
            if self.algorithm:
                self.algorithm.Debug(f"Error in {function_name}: {{e}}")
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
            return f"np.mean({expr}[-{window}:])"

        elif func_name == 'Sum':
            expr, window = self._extract_expr_and_window(args)
            return f"np.sum({expr}[-{window}:])"

        elif func_name == 'Std':
            expr, window = self._extract_expr_and_window(args)
            return f"np.std({expr}[-{window}:])"

        elif func_name == 'Var':
            expr, window = self._extract_expr_and_window(args)
            return f"np.var({expr}[-{window}:])"

        elif func_name == 'Max':
            expr, window = self._extract_expr_and_window(args)
            return f"np.max({expr}[-{window}:])"

        elif func_name == 'Min':
            expr, window = self._extract_expr_and_window(args)
            return f"np.min({expr}[-{window}:])"

        elif func_name == 'Med':
            expr, window = self._extract_expr_and_window(args)
            return f"np.median({expr}[-{window}:])"

        elif func_name == 'Delta':
            expr, window = self._extract_expr_and_window(args)
            return f"({expr}[-1] - {expr}[-{window}] if len({expr}) >= {window} else 0)"

        elif func_name == 'Ref':
            expr, window = self._extract_expr_and_window(args)
            return f"({expr}[-{window}] if len({expr}) >= {window} else np.nan)"

        # Special operators
        elif func_name == 'Mad':
            expr, window = self._extract_expr_and_window(args)
            return f"self._rolling_mad({expr}, {window})[-1]"

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
            expr2 = args[1][1]
            return f"np.corrcoef({expr1}[-{window1}:], {expr2}[-{window1}:])[0, 1] if len({expr1}) >= {window1} else 0"

        elif func_name == 'Cov':
            expr1, window1 = self._extract_expr_and_window([args[0], args[2]])
            expr2 = args[1][1]
            return f"np.cov({expr1}[-{window1}:], {expr2}[-{window1}:])[0, 1] if len({expr1}) >= {window1} else 0"

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

        return expr_code, window_value

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
        return '''    def _wma(self, data, window):
        """Weighted Moving Average"""
        if len(data) < window:
            return np.mean(data)
        weights = np.arange(1, window + 1)
        return np.average(data[-window:], weights=weights)

    def _ema(self, data, window):
        """Exponential Moving Average"""
        if len(data) == 0:
            return np.nan
        if len(data) < window:
            return np.mean(data)

        alpha = 2 / (window + 1)
        ema = data[0]
        for val in data[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema'''


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
            # Generate a fallback function that returns NaN
            fallback_code = f'''    def {func_name}(self, h):
        """{expr}"""
        # CONVERSION FAILED: {str(e)}
        return np.nan'''
            function_codes.append(fallback_code)

    helper_code = ExpressionConverter.generate_helper_functions()

    return function_codes, helper_code


if __name__ == "__main__":
    # Test the converter
    test_expressions = [
        "Mean($close, 20d)",
        "Div(Mean($volume,10d),Greater(Mul(0.5,$low),-1.0))",
        "Mul(Div(Mean(Mad(Div(10.0,Greater($high,-0.01)),40d),40d),-2.0),1.0)",
        "Add($close, $open)",
        "Log(Div(2.0,$high))",
        "Abs(Sub($close, $open))",
    ]

    converter = ExpressionConverter()

    for i, expr in enumerate(test_expressions):
        print(f"\n{'='*80}")
        print(f"Expression {i+1}: {expr}")
        print(f"{'='*80}")
        try:
            code = converter.convert(expr, f"_f{i+1}")
            print(code)
        except Exception as e:
            print(f"ERROR: {e}")

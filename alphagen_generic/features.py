from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType

# ============================================================================
# PRICE/VOLUME FEATURES
# ============================================================================
high = High = HIGH = Feature(FeatureType.HIGH)
low = Low = LOW = Feature(FeatureType.LOW)
volume = Volume = VOLUME = Feature(FeatureType.VOLUME)
open_ = Open = OPEN = Feature(FeatureType.OPEN)
close = Close = CLOSE = Feature(FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(FeatureType.VWAP)

# ============================================================================
# VALUATION RATIOS
# ============================================================================
pe_ratio = PeRatio = PE_RATIO = Feature(FeatureType.PE_RATIO)
pb_ratio = PbRatio = PB_RATIO = Feature(FeatureType.PB_RATIO)
ps_ratio = PsRatio = PS_RATIO = Feature(FeatureType.PS_RATIO)
ev_to_ebitda = EvToEbitda = EV_TO_EBITDA = Feature(FeatureType.EV_TO_EBITDA)
ev_to_revenue = EvToRevenue = EV_TO_REVENUE = Feature(FeatureType.EV_TO_REVENUE)
ev_to_fcf = EvToFcf = EV_TO_FCF = Feature(FeatureType.EV_TO_FCF)

# ============================================================================
# YIELD METRICS
# ============================================================================
earnings_yield = EarningsYield = EARNINGS_YIELD = Feature(FeatureType.EARNINGS_YIELD)
fcf_yield = FcfYield = FCF_YIELD = Feature(FeatureType.FCF_YIELD)
sales_yield = SalesYield = SALES_YIELD = Feature(FeatureType.SALES_YIELD)
dividend_yield = DividendYield = DIVIDEND_YIELD = Feature(FeatureType.DIVIDEND_YIELD)

# ============================================================================
# FORWARD-LOOKING METRICS
# ============================================================================
forward_pe_ratio = ForwardPeRatio = FORWARD_PE_RATIO = Feature(FeatureType.FORWARD_PE_RATIO)

# ============================================================================
# MARKET METRICS
# ============================================================================
shares_outstanding = SharesOutstanding = SHARES_OUTSTANDING = Feature(FeatureType.SHARES_OUTSTANDING)
market_cap = MarketCap = MARKET_CAP = Feature(FeatureType.MARKET_CAP)
turnover = Turnover = TURNOVER = Feature(FeatureType.TURNOVER)

# ============================================================================
# GROWTH METRICS
# ============================================================================
revenue_growth = RevenueGrowth = REVENUE_GROWTH = Feature(FeatureType.REVENUE_GROWTH)
earnings_growth = EarningsGrowth = EARNINGS_GROWTH = Feature(FeatureType.EARNINGS_GROWTH)
book_value_growth = BookValueGrowth = BOOK_VALUE_GROWTH = Feature(FeatureType.BOOK_VALUE_GROWTH)

# ============================================================================
# FINANCIAL HEALTH / LEVERAGE
# ============================================================================
debt_to_assets = DebtToAssets = DEBT_TO_ASSETS = Feature(FeatureType.DEBT_TO_ASSETS)
debt_to_equity = DebtToEquity = DEBT_TO_EQUITY = Feature(FeatureType.DEBT_TO_EQUITY)
current_ratio = CurrentRatio = CURRENT_RATIO = Feature(FeatureType.CURRENT_RATIO)
quick_ratio = QuickRatio = QUICK_RATIO = Feature(FeatureType.QUICK_RATIO)

# ============================================================================
# PROFITABILITY METRICS
# ============================================================================
roe = ROE = Feature(FeatureType.ROE)
roa = ROA = Feature(FeatureType.ROA)
roic = ROIC = Feature(FeatureType.ROIC)
gross_margin = GrossMargin = GROSS_MARGIN = Feature(FeatureType.GROSS_MARGIN)
operating_margin = OperatingMargin = OPERATING_MARGIN = Feature(FeatureType.OPERATING_MARGIN)
net_margin = NetMargin = NET_MARGIN = Feature(FeatureType.NET_MARGIN)

# ============================================================================
# TARGET DEFINITION
# ============================================================================
target = Ref(close, -20) / close - 1

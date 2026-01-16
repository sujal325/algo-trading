import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0  # Added volume for volume-based filters (assuming data provides it)

@dataclass
class Swing:
    high: float
    low: float
    index: int  # Added to track position in candle array

@dataclass
class Zone:
    high: float
    low: float
    touched: bool = False
    strength: float = 1.0  # Added strength based on volume or touches

# =========================
# MARKET STRUCTURE (Enhanced with better swing detection and multi-TF alignment)
# =========================
def detect_swings(candles: List[Candle], period: int = 5) -> List[Swing]:
    """
    Detect swings using a more robust method (zigzag-like with pivot confirmation).
    """
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    swings = []
    for i in range(period, len(candles) - period):
        if highs[i] == np.max(highs[i - period:i + period + 1]):
            swings.append(Swing(high=highs[i], low=0, index=i))  # High swing
        elif lows[i] == np.min(lows[i - period:i + period + 1]):
            swings.append(Swing(high=0, low=lows[i], index=i))  # Low swing
    return swings

def detect_bos(swings: List[Swing]) -> Optional[str]:
    if len(swings) < 3:
        return None
    last_swing = swings[-1]
    prev_swing = swings[-2]
    if last_swing.high > 0 and last_swing.high > prev_swing.high:  # Bull BOS
        return "BULL"
    if last_swing.low > 0 and last_swing.low < prev_swing.low:  # Bear BOS
        return "BEAR"
    return None

def higher_tf_alignment(candles_htf: List[Candle], direction: str) -> bool:
    """
    Check alignment with higher timeframe (e.g., 15m or 1h) trend using simple MA.
    Assume candles_htf is provided as higher TF data.
    """
    closes = np.array([c.close for c in candles_htf[-50:]])  # Last 50 candles
    ma_short = np.mean(closes[-10:])
    ma_long = np.mean(closes[-50:])
    if direction == "BULL":
        return ma_short > ma_long
    if direction == "BEAR":
        return ma_short < ma_long
    return False

def impulse_strength(candle: Candle) -> bool:
    body = abs(candle.close - candle.open)
    range_ = candle.high - candle.low
    return range_ > 0 and body / range_ > 0.6 and candle.volume > np.mean([c.volume for c in candles[-10:]])  # Added volume filter

# =========================
# FIBONACCI ENGINE (Enhanced with dynamic levels and extensions)
# =========================
def fib_zone(high: float, low: float) -> Dict[str, float]:
    diff = high - low
    return {
        "38.2": high - 0.382 * diff,
        "50": high - 0.5 * diff,
        "61.8": high - 0.618 * diff,
        "78.6": high - 0.786 * diff,
        "88.6": high - 0.886 * diff,  # Added deeper level
    }

def in_deep_pullback(price: float, fib: Dict[str, float], direction: str) -> bool:
    if direction == "BULL":
        return fib["88.6"] <= price <= fib["50"]
    if direction == "BEAR":
        return fib["50"] <= price <= fib["88.6"]
    return False

def tp_extensions(high: float, low: float, direction: str) -> Dict[str, float]:
    move = abs(high - low)
    if direction == "BULL":
        return {
            "1.618": high + 1.618 * move,
            "2.618": high + 2.618 * move,  # Added further extension
        }
    if direction == "BEAR":
        return {
            "1.618": low - 1.618 * move,
            "2.618": low - 2.618 * move,
        }
    return {}

# =========================
# LIQUIDITY ENGINE (Enhanced with volume confirmation and multiple lookbacks)
# =========================
def liquidity_sweep(candles: List[Candle], direction: str, lookbacks: List[int] = [5, 10, 20]) -> bool:
    for lb in lookbacks:
        recent = candles[-lb:]
        if direction == "BULL":
            if candles[-2].low < min(c.low for c in recent[:-1]) and candles[-1].volume > np.mean([c.volume for c in recent]):
                return True
        if direction == "BEAR":
            if candles[-2].high > max(c.high for c in recent[:-1]) and candles[-1].volume > np.mean([c.volume for c in recent]):
                return True
    return False

# =========================
# SUPPLY & DEMAND (Enhanced with zone strength and invalidation)
# =========================
def fresh_zone(zone: Zone, price: float) -> bool:
    if zone.touched:
        return False
    in_zone = zone.low <= price <= zone.high
    if in_zone:
        zone.touched = True
        zone.strength -= 0.2  # Decay strength on touch
    return in_zone and zone.strength > 0.5  # Only consider strong zones

# =========================
# VOLATILITY / ATR (Vectorized for efficiency)
# =========================
def atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    return np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)

def rolling_atr(candles: List[Candle], window: int = 10, period: int = 14) -> np.ndarray:
    """
    Compute rolling ATR for efficiency in averages.
    """
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
    rolling = np.convolve(tr, np.ones(period) / period, mode='valid')
    return rolling[-window:]

# =========================
# MARKET CYCLE ENGINE (Enhanced with more states and volume)
# =========================
def market_cycle(bos: Optional[str], atr_now: float, atr_avg: float, impulse: bool, volume_trend: float) -> str:
    if bos and atr_now > 1.2 * atr_avg and impulse and volume_trend > 1.0:
        return "STRONG_EXPANSION"  # New state for high confidence
    if bos and atr_now > atr_avg and impulse:
        return "EXPANSION"
    elif bos and atr_now <= atr_avg:
        return "MATURITY"
    elif not bos and atr_now > atr_avg:
        return "DISTRIBUTION"
    else:
        return "CHOP"

def volume_trend(candles: List[Candle], period: int = 10) -> float:
    volumes = np.array([c.volume for c in candles[-period * 2:]])
    return np.mean(volumes[-period:]) / np.mean(volumes[:-period]) if np.mean(volumes[:-period]) > 0 else 1.0

def risk_by_cycle(cycle: str) -> float:
    return {
        "STRONG_EXPANSION": 0.015,  # Higher risk for high confidence
        "EXPANSION": 0.01,
        "MATURITY": 0.005,
        "DISTRIBUTION": 0.0025,
        "CHOP": 0.0
    }[cycle]

# =========================
# CAPITAL ENGINE (Enhanced with drawdown protection)
# =========================
def effective_equity(balance: float, equity: float, max_drawdown: float = 0.05) -> float:
    eq = max(balance, equity)
    if (balance - equity) / balance > max_drawdown:
        return 0.0  # Halt trading on excessive drawdown
    return eq

# =========================
# DYNAMIC STOP SYSTEM (Enhanced with trailing and breakeven)
# =========================
def dynamic_sl(entry: float, current_price: float, r_multiple: float, last_internal_swing: float, atr_val: float, direction: str) -> float:
    if r_multiple >= 3:
        # Trail with ATR
        if direction == "BULL":
            return current_price - 1.5 * atr_val
        else:
            return current_price + 1.5 * atr_val
    elif r_multiple >= 2:
        return last_internal_swing
    elif r_multiple >= 1:
        return entry  # Breakeven
    return entry - atr_val if direction == "BULL" else entry + atr_val  # Initial SL

# =========================
# TRADE GOVERNANCE (Enhanced with time-based limits and win/loss streak)
# =========================
class TradeManager:
    def __init__(self):
        self.trades_today = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_trades_day = 5  # Configurable

    def allow_trade(self, cycle: str, is_win: bool = None) -> bool:
        if is_win is not None:
            if is_win:
                self.win_streak += 1
                self.loss_streak = 0
            else:
                self.loss_streak += 1
                self.win_streak = 0
        if self.loss_streak >= 3:
            return False  # Halt on losing streak
        if cycle == "DISTRIBUTION":
            return self.trades_today < 1
        if cycle == "CHOP":
            return False
        return self.trades_today < self.max_trades_day + self.win_streak  # Allow more on wins

    def register_trade(self):
        self.trades_today += 1

# =========================
# MAIN DECISION ENGINE (Enhanced with weighted confluence, HTF, and efficiency)
# =========================
def evaluate_trade(
    candles_5m: List[Candle],
    candles_1m: List[Candle],
    candles_htf: List[Candle],  # Added higher TF
    zones: List[Zone],  # Now list of zones
    balance: float,
    equity: float,
    last_internal_swing: float,
    trade_manager: TradeManager
) -> Optional[Dict[str, any]]:
    swings = detect_swings(candles_5m)  # Now detect swings internally
    direction = detect_bos(swings)
    if not direction or not higher_tf_alignment(candles_htf, direction):
        return None

    impulse = impulse_strength(candles_5m[-1])
    atr_now = atr(candles_5m)
    rolling_atrs = rolling_atr(candles_5m, window=10)
    atr_avg = np.mean(rolling_atrs)
    vol_trend = volume_trend(candles_5m)
    cycle = market_cycle(direction, atr_now, atr_avg, impulse, vol_trend)
    risk = risk_by_cycle(cycle)
    if risk == 0 or not trade_manager.allow_trade(cycle):
        return None

    price = candles_1m[-1].close
    swing_high = max(s.high for s in swings if s.high > 0)
    swing_low = min(s.low for s in swings if s.low > 0)
    fib = fib_zone(swing_high, swing_low)

    # Weighted confluence
    confluence_score = 0.0
    if in_deep_pullback(price, fib, direction):
        confluence_score += 2.0  # High weight for deep pullback
    for zone in zones:
        if fresh_zone(zone, price):
            confluence_score += zone.strength  # Weighted by zone strength
            break
    if liquidity_sweep(candles_1m, direction):
        confluence_score += 1.5
    if impulse:
        confluence_score += 1.0
    if direction:
        confluence_score += 0.5  # Base for direction

    min_confluence = 5.0 if cycle == "STRONG_EXPANSION" else 6.0 if cycle == "EXPANSION" else 7.0
    if confluence_score < min_confluence:
        return None

    entry = price
    extensions = tp_extensions(swing_high, swing_low, direction)
    tp = extensions["1.618"]  # Primary TP, can partial at 1.618 and hold to 2.618
    stop = dynamic_sl(entry, price, 0, last_internal_swing, atr_now, direction)  # Initial SL
    rr = abs(tp - entry) / abs(entry - stop)
    if rr < 4:
        return None

    eff_eq = effective_equity(balance, equity)
    size = eff_eq * risk / abs(entry - stop)  # Position size based on risk per trade (enhanced)

    trade_manager.register_trade()

    return {
        "direction": direction,
        "cycle": cycle,
        "entry": entry,
        "stop": stop,
        "tp": tp,
        "tp_extensions": extensions,
        "risk_pct": risk,
        "position_size": size,
        "rr": rr,
        "confluence_score": confluence_score
    }
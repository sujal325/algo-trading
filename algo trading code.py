import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: Optional[float] = None

class InstitutionalScalper:
    def __init__(self):
        self.trades_today = deque(maxlen=24)
        self.daily_loss = 0.0
        
    def find_swings(self, df: pd.DataFrame, period: int = 5) -> Tuple[List[float], List[float]]:
        """Detect swing highs/lows for market structure"""
        highs = []
        lows = []
        
        for i in range(period, len(df) - period):
            if df['high'].iloc[i] == df['high'].iloc[i-period:i+period+1].max():
                highs.append((df['high'].iloc[i], i))
            if df['low'].iloc[i] == df['low'].iloc[i-period:i+period+1].min():
                lows.append((df['low'].iloc[i], i))
                
        return highs[-8:], lows[-8:]  # Last 8 swings
    
    def detect_bos(self, highs: List[Tuple[float, int]], lows: List[Tuple[float, int]]) -> Dict:
        """Break of Structure - Institutional trend change"""
        if len(highs) >= 3 and highs[-1][0] > highs[-2][0] > highs[-3][0]:
            return {"direction": "BULL", "level": highs[-1][0], "confirmed": True}
        if len(lows) >= 3 and lows[-1][0] < lows[-2][0] < lows[-3][0]:
            return {"direction": "BEAR", "level": lows[-1][0], "confirmed": True}
        return {"direction": None}
    
    def find_supply_demand(self, df: pd.DataFrame) -> List[Dict]:
        """Order Blocks - Last opposing candle before strong move"""
        strong_moves = df['close'].diff(5) > df['close'].diff(5).std() * 1.5
        zones = []
        
        for i in strong_moves[strong_moves].index:
            if df['close'].iloc[i] > df['close'].iloc[i-5]:
                # Bullish OB (demand)
                zones.append({
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'type': 'demand',
                    'strength': df['volume'].iloc[i-1] / df['volume'].tail(20).mean()
                })
            else:
                # Bearish OB (supply)  
                zones.append({
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'type': 'supply',
                    'strength': df['volume'].iloc[i-1] / df['volume'].tail(20).mean()
                })
        return zones[-5:]  # Recent 5 zones
    
    def liquidity_zones(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Equal highs/lows = stop hunt zones"""
        highs = df['high'].tail(30)
        lows = df['low'].tail(30)
        
        # Most touched levels
        top_liq = highs.mode().iloc[0] if len(highs.mode()) > 0 else highs.max()
        bottom_liq = lows.mode().iloc[0] if len(lows.mode()) > 0 else lows.min()
        
        return top_liq, bottom_liq
    
    def fib_levels(self, swing_high: float, swing_low: float, direction: str) -> Dict:
        """Deep pullbacks only (61.8-78.6%) + Extensions"""
        range_ = swing_high - swing_low
        
        retracement = {
            "618": swing_high - 0.618 * range_,
            "786": swing_high - 0.786 * range_
        }
        
        extension = {
            "1272": swing_low + 1.272 * range_ if direction == "BULL" else swing_high - 1.272 * range_,
            "1618": swing_low + 1.618 * range_ if direction == "BULL" else swing_high - 1.618 * range_
        }
        
        return {"retracement": retracement, "extension": extension}
    
    def internal_structure(self, df: pd.DataFrame) -> bool:
        """Internal range liquidity before breakout"""
        recent_range = df['high'].tail(10).max() - df['low'].tail(10).min()
        current_price = df['close'].iloc[-1]
        range_middle = df['low'].tail(10).min() + recent_range * 0.5
        
        # Price swept internal liquidity (50% level)
        return abs(current_price - range_middle) < recent_range * 0.1
    
    def check_professional_setup(self, candles: List[Candle], balance: float) -> Optional[Dict]:
        """Institutional-grade entry logic"""
        if len(candles) < 50:
            return None
            
        df = pd.DataFrame({
            'open': [c.open for c in candles],
            'high': [c.high for c in candles], 
            'low': [c.low for c in candles],
            'close': [c.close for c in candles],
            'volume': [c.volume for c in candles]
        })
        
        current = df.iloc[-1]
        atr = pd.Series(df['high'] - df['low']).tail(14).mean()
        
        # 1. MARKET STRUCTURE (BOS confirmed)
        highs, lows = self.find_swings(df)
        bos = self.detect_bos(highs, lows)
        if not bos['confirmed']:
            return None
            
        # 2. LIQUIDITY SWEEP (price grabbed stops)
        liq_high, liq_low = self.liquidity_zones(df)
        swept_low = current['low'] <= liq_low * 1.001  # 0.1 pip tolerance
        swept_high = current['high'] >= liq_high * 0.999
        
        liquidity_grab = (bos['direction'] == "BULL" and swept_low) or \
                        (bos['direction'] == "BEAR" and swept_high)
        if not liquidity_grab:
            return None
        
        # 3. SUPPLY/DEMAND ZONE
        zones = self.find_supply_demand(df)
        valid_zone = next((z for z in zones 
                          if (z['type'] == 'demand' and bos['direction'] == "BULL") or 
                            (z['type'] == 'supply' and bos['direction'] == "BEAR") and
                            z['strength'] > 1.5), None)
        
        if not valid_zone:
            return None
            
        # 4. DEEP FIBO PULLBACK ONLY (61.8-78.6%)
        swing_high = max([h[0] for h in highs[-3:]])
        swing_low = min([l[0] for l in lows[-3:]])
        fib = self.fib_levels(swing_high, swing_low, bos['direction'])
        
        current_price = current['close']
        in_fib_zone = False
        
        if bos['direction'] == "BULL":
            in_fib_zone = fib['retracement']['786'] <= current_price <= fib['retracement']['618']
        else:
            in_fib_zone = fib['retracement']['618'] <= current_price <= fib['retracement']['786']
            
        if not in_fib_zone:
            return None
        
        # 5. INTERNAL STRUCTURE + ORDER BLOCK TOUCH
        if not self.internal_structure(df):
            return None
            
        # 6. HIGH R:R CALCULATION (Min 1:4)
        entry = current_price
        stop = valid_zone['low'] if bos['direction'] == "BULL" else valid_zone['high']
        target = fib['extension']['1272']
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk
        
        if rr < 4.0:  # Minimum 1:4
            return None
        
        # Position sizing (0.5% risk)
        pip_value = 0.10  # 0.01 lot EURUSD
        lot_size = min(0.01, (balance * 0.005) / (risk / 0.0001 * pip_value))
        
        return {
            "direction": bos['direction'],
            "entry": round(entry, 5),
            "stop": round(stop, 5),
            "target": round(target, 5),
            "lot_size": lot_size,
            "rr": round(rr, 2),
            "fib_level": "61.8-78.6%",
            "zone_type": valid_zone['type'],
            "structure": "BOS confirmed",
            "atr": round(atr, 5)
        }

# === USAGE ===
def run_institutional_scalper(candles: List[Candle], balance: float = 100.0):
    scalper = InstitutionalScalper()
    signal = scalper.check_professional_setup(candles, balance)
    
    if signal:
        print("🏦 INSTITUTIONAL SETUP:")
        for k, v in signal.items():
            print(f"{k}: {v}")
        return signal
    print("No institutional setup")
    return None

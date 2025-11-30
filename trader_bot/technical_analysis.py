import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Calculate technical indicators for trading decisions."""
    
    def __init__(self):
        pass
    
    def calculate_indicators(self, market_data: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """
        Calculate technical indicators from market data.
        
        Args:
            market_data: List of market data dicts (most recent first)
        
        Returns:
            Dict of indicators or None if insufficient data
        """
        if len(market_data) < 20:
            logger.warning(f"Insufficient data for indicators: {len(market_data)} points (need 20+)")
            return None
        
        # Convert to DataFrame (reverse to chronological order)
        df = pd.DataFrame(reversed(market_data))
        df['price'] = pd.to_numeric(df['price'])
        
        indicators = {}
        
        try:
            # RSI (14-period)
            indicators['rsi'] = self._calculate_rsi(df['price'], period=14)
            
            # MACD
            macd_data = self._calculate_macd(df['price'])
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(df['price'], period=20)
            indicators['bb_upper'] = bb_data['upper']
            indicators['bb_middle'] = bb_data['middle']
            indicators['bb_lower'] = bb_data['lower']
            indicators['bb_width'] = bb_data['width']
            
            # Simple Moving Averages
            indicators['sma_20'] = df['price'].rolling(window=20).mean().iloc[-1]
            if len(df) >= 50:
                indicators['sma_50'] = df['price'].rolling(window=50).mean().iloc[-1]

            if any(pd.isna(val) or np.isinf(val) for val in indicators.values()):
                logger.warning("Indicator calculation produced invalid values; skipping snapshot")
                return None
            
            logger.debug(f"Calculated indicators: RSI={indicators['rsi']:.2f}, MACD={indicators['macd']:.4f}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        middle = sma.iloc[-1]
        width = ((upper_band.iloc[-1] - lower_band.iloc[-1]) / middle * 100) if middle else float("nan")
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': middle,
            'lower': lower_band.iloc[-1],
            'width': width
        }
    
    def get_trading_signals(self, indicators: Dict[str, float], current_price: float) -> Dict[str, str]:
        """
        Generate trading signals from indicators.
        
        Returns:
            Dict with signal interpretations
        """
        if not indicators:
            return {}
        
        signals = {}
        
        # RSI signals
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            signals['rsi'] = "OVERBOUGHT - Consider selling"
        elif rsi < 30:
            signals['rsi'] = "OVERSOLD - Consider buying"
        else:
            signals['rsi'] = "NEUTRAL"
        
        # MACD signals
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            signals['macd'] = "BULLISH - MACD above signal"
        else:
            signals['macd'] = "BEARISH - MACD below signal"
        
        # Bollinger Bands signals
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        if bb_upper and bb_lower:
            if current_price >= bb_upper:
                signals['bollinger'] = "OVERBOUGHT - Price at upper band"
            elif current_price <= bb_lower:
                signals['bollinger'] = "OVERSOLD - Price at lower band"
            else:
                signals['bollinger'] = "NEUTRAL - Price in normal range"
        
        # Moving average signals
        sma_20 = indicators.get('sma_20')
        if sma_20:
            if current_price > sma_20:
                signals['trend'] = "UPTREND - Price above SMA20"
            else:
                signals['trend'] = "DOWNTREND - Price below SMA20"
        
        return signals
    
    def format_indicators_for_llm(self, indicators: Dict[str, float], current_price: float) -> str:
        """Format indicators as a readable string for LLM context."""
        if not indicators:
            return "No technical indicators available (insufficient data)"
        
        signals = self.get_trading_signals(indicators, current_price)
        
        output = "Technical Indicators:\n"
        output += f"  RSI (14): {indicators.get('rsi', 0):.2f} - {signals.get('rsi', 'N/A')}\n"
        output += f"  MACD: {indicators.get('macd', 0):.4f} (Signal: {indicators.get('macd_signal', 0):.4f}) - {signals.get('macd', 'N/A')}\n"
        output += f"  Bollinger Bands: ${indicators.get('bb_lower', 0):,.2f} - ${indicators.get('bb_upper', 0):,.2f} - {signals.get('bollinger', 'N/A')}\n"
        output += f"  SMA (20): ${indicators.get('sma_20', 0):,.2f} - {signals.get('trend', 'N/A')}\n"
        
        return output

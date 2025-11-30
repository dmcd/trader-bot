import unittest
from datetime import datetime, timedelta

from trader_bot.technical_analysis import TechnicalAnalysis


class TestTechnicalAnalysis(unittest.TestCase):
    def setUp(self):
        self.ta = TechnicalAnalysis()

    def _build_series(self, n=30, start=100.0, step=1.0):
        now = datetime.now()
        data = []
        for i in range(n):
            price = start + (i * step)
            data.append(
                {
                    "timestamp": (now + timedelta(seconds=i)).isoformat(),
                    "price": price,
                    "bid": price - 0.5,
                    "ask": price + 0.5,
                }
            )
        return list(reversed(data))  # Most recent first, as expected by calculate_indicators

    def test_requires_minimum_history(self):
        indicators = self.ta.calculate_indicators(self._build_series(n=10))
        self.assertIsNone(indicators)

    def test_indicator_fields_present(self):
        indicators = self.ta.calculate_indicators(self._build_series(n=40))
        self.assertIsNotNone(indicators)
        for key in ["rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "sma_20", "bb_width"]:
            self.assertIn(key, indicators)

        signals = self.ta.get_trading_signals(indicators, current_price=110.0)
        self.assertIn("rsi", signals)

        formatted = self.ta.format_indicators_for_llm(indicators, current_price=110.0)
        self.assertIn("Technical Indicators:", formatted)

    def test_nan_inputs_are_rejected(self):
        indicators = self.ta.calculate_indicators(self._build_series(n=25, start=0.0, step=0.0))
        self.assertIsNone(indicators)

    def test_trading_signals_cover_all_branches(self):
        bullish_indicators = {
            "rsi": 80,
            "macd": 1.0,
            "macd_signal": 0.5,
            "bb_upper": 110.0,
            "bb_lower": 90.0,
            "sma_20": 100.0,
        }
        bullish = self.ta.get_trading_signals(bullish_indicators, current_price=110.0)
        self.assertEqual(bullish["rsi"], "OVERBOUGHT - Consider selling")
        self.assertEqual(bullish["macd"], "BULLISH - MACD above signal")
        self.assertEqual(bullish["bollinger"], "OVERBOUGHT - Price at upper band")
        self.assertEqual(bullish["trend"], "UPTREND - Price above SMA20")

        bearish_indicators = {
            "rsi": 20,
            "macd": -1.0,
            "macd_signal": -0.5,
            "bb_upper": 110.0,
            "bb_lower": 90.0,
            "sma_20": 100.0,
        }
        bearish = self.ta.get_trading_signals(bearish_indicators, current_price=80.0)
        self.assertEqual(bearish["rsi"], "OVERSOLD - Consider buying")
        self.assertEqual(bearish["macd"], "BEARISH - MACD below signal")
        self.assertEqual(bearish["bollinger"], "OVERSOLD - Price at lower band")
        self.assertEqual(bearish["trend"], "DOWNTREND - Price below SMA20")


if __name__ == "__main__":
    unittest.main()

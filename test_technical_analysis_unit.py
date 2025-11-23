import unittest
from datetime import datetime, timedelta

from technical_analysis import TechnicalAnalysis


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


if __name__ == "__main__":
    unittest.main()

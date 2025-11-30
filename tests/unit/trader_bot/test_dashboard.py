import importlib
import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest


class DummyContext:
    def __init__(self):
        self.column_config = types.SimpleNamespace(
            NumberColumn=lambda *args, **kwargs: None,
            TextColumn=lambda *args, **kwargs: None,
            DatetimeColumn=lambda *args, **kwargs: None,
        )
        self.sidebar = self

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, _name):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    # Streamlit-style helpers
    def tabs(self, labels):
        return [self for _ in labels]

    def columns(self, labels):
        return [self for _ in labels]

    def button(self, *args, **kwargs):
        return False

    def dataframe(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def rerun(self):
        return None

    def set_page_config(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None


class FakeDB:
    def __init__(self):
        self.conn = None

    def get_session_id_by_version(self, _version):
        return None

    def list_bot_versions(self):
        return []

    def close(self):
        return None

    def get_health_state(self):
        return []

    def get_session_stats(self, _session_id):
        return {}


def load_dashboard(monkeypatch, tmp_path):
    sys.modules.pop("trader_bot.dashboard", None)
    dummy_streamlit = DummyContext()
    fake_db_module = types.SimpleNamespace(TradingDatabase=FakeDB)
    fake_subprocess = types.SimpleNamespace(
        DEVNULL=None,
        run=lambda *args, **kwargs: types.SimpleNamespace(stdout=""),
        Popen=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "streamlit", dummy_streamlit)
    monkeypatch.setitem(sys.modules, "trader_bot.database", fake_db_module)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("trader_bot.dashboard")
    return module


def test_format_ratio_badge_and_health_summary(monkeypatch, tmp_path):
    dashboard = load_dashboard(monkeypatch, tmp_path)

    assert "High" in dashboard.format_ratio_badge(30)
    assert dashboard.format_ratio_badge(None) is None
    summary = dashboard.summarize_health_detail('{"circuit":"open","note":null}')
    assert "circuit: open" in summary
    assert dashboard.summarize_health_detail("") == ""


def test_calculate_pnl_shapes_positions(monkeypatch, tmp_path):
    dashboard = load_dashboard(monkeypatch, tmp_path)
    ts0 = datetime.now(timezone.utc)
    ts1 = ts0 + timedelta(seconds=60)
    df = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTC/USD", "action": "BUY", "price": 100.0, "quantity": 1.0},
            {"timestamp": ts1, "symbol": "BTC/USD", "action": "SELL", "price": 120.0, "quantity": 1.0},
        ]
    )

    realized, unrealized, positions, shaped_df, exposure, spacing = dashboard.calculate_pnl(
        df, {"BTC/USD": 125.0}
    )

    assert realized == pytest.approx(20.0)
    assert unrealized == pytest.approx(0.0)
    assert positions == []
    assert shaped_df["pnl"].iloc[-1] == pytest.approx(20.0)
    assert spacing["last_seconds"] == pytest.approx(60.0)
    assert exposure == pytest.approx(0.0)

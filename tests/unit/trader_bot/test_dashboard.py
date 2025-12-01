import importlib
import sys
import types
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

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

    def selectbox(self, _label=None, options=None, index=0, format_func=None):
        options = options or []
        if not options:
            return None
        try:
            choice = options[index]
        except Exception:
            choice = options[0]
        if format_func:
            try:
                format_func(choice)
            except Exception:
                pass
        return choice

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

    def get_portfolio_id_by_version(self, _version):
        return None

    def list_bot_versions(self):
        return []

    def list_portfolios(self, bot_version=None):
        return []

    def close(self):
        return None

    def get_health_state(self):
        return []

    def get_portfolio_stats(self, _portfolio_id):
        return {}

    def get_portfolio(self, _portfolio_id):
        return {}

    def get_latest_run_metadata_for_portfolio(self, _portfolio_id):
        return {}

    def get_trades_for_portfolio(self, _portfolio_id):
        return []


@pytest.fixture(scope="module")
def dashboard_module(tmp_path_factory):
    """Load the dashboard module once with patched dependencies to avoid repeated imports."""
    tmp_path = tmp_path_factory.mktemp("dashboard_module")
    monkeypatch = pytest.MonkeyPatch()
    dummy_streamlit = DummyContext()
    fake_db_module = types.SimpleNamespace(TradingDatabase=FakeDB)
    fake_subprocess = types.SimpleNamespace(
        DEVNULL=None,
        run=lambda *args, **kwargs: types.SimpleNamespace(stdout=""),
        Popen=lambda *args, **kwargs: None,
    )

    sys.modules.pop("trader_bot.dashboard", None)
    monkeypatch.setitem(sys.modules, "streamlit", dummy_streamlit)
    monkeypatch.setitem(sys.modules, "trader_bot.database", fake_db_module)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module("trader_bot.dashboard")
    yield module

    monkeypatch.undo()
    sys.modules.pop("trader_bot.dashboard", None)


def test_format_ratio_badge_and_health_summary(dashboard_module):
    dashboard = dashboard_module

    assert "High" in dashboard.format_ratio_badge(30)
    assert dashboard.format_ratio_badge(None) is None
    summary = dashboard.summarize_health_detail('{"circuit":"open","note":null}')
    assert "circuit: open" in summary
    assert dashboard.summarize_health_detail("") == ""


def test_venue_status_payload_and_badges(dashboard_module):
    dashboard = dashboard_module
    now = datetime(2024, 6, 3, 1, 0, tzinfo=timezone.utc)  # Monday 11:00am AEST
    health = [
        {"key": "exchange_circuit", "value": "tripped", "detail": '{"context":"md"}', "updated_at": "2024-06-03T01:00:00Z"}
    ]
    payload = dashboard.build_venue_status_payload(
        "IB",
        {"base_currency": "AUD"},
        {"base_currency": "NZD"},
        health,
        now=now,
    )
    assert payload["base_currency"] == "AUD"
    hours = payload.get("market_hours")
    assert any(entry["label"] == "ASX cash" and entry["is_open"] for entry in hours)
    circuit = payload.get("circuit", {}).get("exchange_circuit")
    assert circuit["status"] == "tripped"
    badge_html = dashboard.format_venue_badge(payload["venue"], payload["base_currency"])
    assert "Venue" in badge_html and "AUD" in badge_html


def test_ib_market_hours_weekend_gap(dashboard_module):
    dashboard = dashboard_module
    friday_close = datetime(2024, 6, 7, 22, 0, tzinfo=timezone.utc)
    sunday_closed = datetime(2024, 6, 9, 12, 0, tzinfo=timezone.utc)
    monday_open = datetime(2024, 6, 10, 0, 0, tzinfo=timezone.utc)
    friday_hours = dashboard.ib_market_hours_status(friday_close)
    sunday_hours = dashboard.ib_market_hours_status(sunday_closed)
    monday_hours = dashboard.ib_market_hours_status(monday_open)
    assert any(entry["label"] == "FX (~24/5)" and entry["is_open"] is False for entry in friday_hours)
    assert any(entry["label"] == "FX (~24/5)" and entry["is_open"] is False for entry in sunday_hours)
    assert any(entry["label"] == "FX (~24/5)" and entry["is_open"] is True for entry in monday_hours)


def test_calculate_pnl_shapes_positions(dashboard_module):
    dashboard = dashboard_module
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


def test_calculate_pnl_partial_and_unrealized(dashboard_module):
    dashboard = dashboard_module
    ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {"timestamp": ts0, "symbol": "BTC/USD", "action": "BUY", "price": 100.0, "quantity": 1.0},
            {"timestamp": ts0 + timedelta(seconds=30), "symbol": "BTC/USD", "action": "BUY", "price": 120.0, "quantity": 0.5},
            {"timestamp": ts0 + timedelta(seconds=120), "symbol": "BTC/USD", "action": "SELL", "price": 130.0, "quantity": 0.3},
            {"timestamp": ts0 + timedelta(seconds=240), "symbol": "ETH/USD", "action": "BUY", "price": 50.0, "quantity": 2.0},
        ]
    )

    realized, unrealized, positions, shaped, exposure, spacing = dashboard.calculate_pnl(
        df, {"BTC/USD": 140.0, "ETH/USD": 55.0}
    )

    assert realized == pytest.approx(7.0, rel=1e-3)
    assert unrealized == pytest.approx(50.0, rel=1e-3)
    assert exposure == pytest.approx((140 * 1.2) + (55 * 2), rel=1e-3)
    assert len(positions) == 2
    assert spacing["avg_seconds"] == pytest.approx(80.0)
    assert spacing["last_seconds"] == pytest.approx(120.0)
    assert shaped["pnl"].iloc[2] == pytest.approx(7.0, rel=1e-3)


def test_ratio_badge_thresholds(dashboard_module):
    dashboard = dashboard_module
    moderate = dashboard.format_ratio_badge(15)
    good = dashboard.format_ratio_badge(5)
    assert "Moderate" in moderate
    assert "Good" in good


def test_timezone_resolution_and_labels(monkeypatch, dashboard_module):
    dashboard = dashboard_module
    monkeypatch.setenv("LOCAL_TIMEZONE", "UTC")
    tz = dashboard.get_user_timezone()
    assert isinstance(tz, ZoneInfo)
    assert dashboard.get_timezone_label(tz) == "UTC"

    offset_tz = timezone(timedelta(hours=-8))
    assert dashboard.get_timezone_label(offset_tz) == offset_tz.tzname(None)

    class BadTZ:
        def tzname(self, dt):
            raise RuntimeError("boom")

    assert dashboard.get_timezone_label(BadTZ()) == "Local"


def test_is_bot_running_and_start_bot_paths(monkeypatch, dashboard_module):
    dashboard = dashboard_module
    called = {}

    class Recorder(types.SimpleNamespace):
        def error(self, msg):
            called["error"] = msg

    monkeypatch.setattr(dashboard, "st", Recorder())
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(stdout="123\n"),
    )
    assert dashboard.is_bot_running() is True

    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    assert dashboard.is_bot_running() is False

    popen_calls = {}

    def fake_popen(cmd, **kwargs):
        popen_calls["cmd"] = cmd
        popen_calls["kwargs"] = kwargs
        return True

    monkeypatch.setattr(dashboard.subprocess, "Popen", fake_popen)
    assert dashboard.start_bot() is True
    assert popen_calls["cmd"][:3] == ['python', '-m', 'trader_bot.strategy_runner']

    def boom(*_args, **_kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(dashboard.subprocess, "Popen", boom)
    assert dashboard.start_bot() is False
    assert "error" in called


def test_load_history_and_prices_handle_empty_and_errors(monkeypatch, dashboard_module):
    dashboard = dashboard_module
    assert dashboard.load_history(None, None).empty
    assert dashboard.get_latest_prices(1, []) == {}

    errors = []

    class Recorder(types.SimpleNamespace):
        def error(self, msg):
            errors.append(msg)

    monkeypatch.setattr(dashboard, "st", Recorder())

    class BrokenDB:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("db down")

    monkeypatch.setattr(dashboard, "TradingDatabase", BrokenDB)
    assert dashboard.load_history(1, None).empty
    assert dashboard.get_latest_prices(1, ["BTC/USD"]) == {}
    assert errors, "expected error messages to be recorded"


def test_load_history_and_prices_success(monkeypatch, dashboard_module):
    dashboard = dashboard_module

    class StubDB:
        def __init__(self):
            self.closed = False
            self._trades = [
                {
                    "timestamp": "2024-01-01T00:00:00+00:00",
                    "symbol": "BTC/USD",
                    "action": "BUY",
                    "price": 100.0,
                    "quantity": 1.0,
                    "fee": 0.1,
                    "liquidity": "maker",
                    "realized_pnl": 5.0,
                    "reason": "test",
                }
            ]
            self._prices = {"BTC/USD": 105.0}

        def close(self):
            self.closed = True

        def get_trades_for_portfolio(self, _portfolio_id):
            return self._trades

        def get_recent_market_data_for_portfolio(self, _portfolio_id, symbol, limit=1):
            price = self._prices.get(symbol)
            if price is None:
                return []
            return [{"price": price}]

    monkeypatch.setattr(dashboard, "TradingDatabase", StubDB)
    df = dashboard.load_history(1, ZoneInfo("UTC"))
    assert not df.empty
    assert "trade_value" in df.columns
    prices = dashboard.get_latest_prices(1, ["BTC/USD"])
    assert prices["BTC/USD"] == 105.0

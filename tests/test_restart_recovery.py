import pytest

from trader_bot.strategy_runner import StrategyRunner


class StubBot:
    def __init__(self, positions=None, orders=None, raise_positions=False):
        self.positions = positions or []
        self.orders = orders or []
        self.raise_positions = raise_positions

    async def get_positions_async(self):
        if self.raise_positions:
            raise RuntimeError("positions fail")
        return self.positions

    async def get_open_orders_async(self):
        return self.orders


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "restart.db"))
    r = StrategyRunner(execute_orders=False)
    r.session_id = r.db.get_or_create_session(starting_balance=1000.0, bot_version="test")
    return r


@pytest.mark.asyncio
async def test_reconcile_replaces_snapshots_and_updates_health(runner):
    runner.bot = StubBot(
        positions=[{"symbol": "BTC/USD", "quantity": 1.0, "avg_price": 10000.0}],
        orders=[
            {
                "order_id": "1",
                "symbol": "BTC/USD",
                "side": "buy",
                "price": 10100,
                "amount": 0.5,
                "remaining": 0.5,
                "client_order_id": "BOT-v1-abc",
            }
        ],
    )
    # seed stale snapshots
    runner.db.replace_positions(runner.session_id, [{"symbol": "ETH/USD", "quantity": 2.0, "avg_price": 2000.0}])
    runner.db.replace_open_orders(
        runner.session_id,
        [{"order_id": "old", "symbol": "ETH/USD", "side": "sell", "price": 2100, "amount": 1, "remaining": 1}],
    )

    await runner._reconcile_exchange_state()

    positions = runner.db.get_positions(runner.session_id)
    orders = runner.db.get_open_orders(runner.session_id)
    assert len(positions) == 1
    assert positions[0]["symbol"] == "BTC/USD"
    assert len(orders) == 1
    assert orders[0]["symbol"] == "BTC/USD"

    health = {row["key"]: row for row in runner.db.get_health_state()}
    assert health["restart_recovery"]["value"] == "ok"


@pytest.mark.asyncio
async def test_reconcile_records_error_on_failure(runner):
    runner.bot = StubBot(raise_positions=True)
    await runner._reconcile_exchange_state()
    health = {row["key"]: row for row in runner.db.get_health_state()}
    assert health["restart_recovery"]["value"] == "error"

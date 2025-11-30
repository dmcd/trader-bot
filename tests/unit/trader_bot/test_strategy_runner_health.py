from trader_bot.strategy_runner import StrategyRunner


def test_record_health_state_includes_portfolio_and_run():
    emitted = []

    class StubDB:
        def __init__(self):
            self.saved = []

        def set_health_state(self, key, value, detail_str=None):
            self.saved.append((key, value, detail_str))

    runner = object.__new__(StrategyRunner)
    runner.db = StubDB()
    runner.session_id = 42
    runner.portfolio_id = 7
    runner.run_id = "run-xyz"
    runner._emit_telemetry = lambda record: emitted.append(record)

    runner._record_health_state("risk", "ok", {"foo": "bar"})

    assert runner.db.saved[-1][0] == "risk"
    assert emitted[0]["session_id"] == 42
    assert emitted[0]["portfolio_id"] == 7
    assert emitted[0]["run_id"] == "run-xyz"

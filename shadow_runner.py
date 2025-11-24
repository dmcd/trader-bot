import asyncio

from strategy_runner import StrategyRunner


async def main():
    runner = StrategyRunner(execute_orders=False)
    await runner.run_loop()


if __name__ == "__main__":
    asyncio.run(main())

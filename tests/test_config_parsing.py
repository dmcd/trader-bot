import importlib
import os
from unittest import mock

import trader_bot.config as config


def reload_config(env: dict[str, str]):
    with mock.patch("dotenv.load_dotenv"), mock.patch.dict(os.environ, env, clear=True):
        return importlib.reload(config)


def test_parse_maker_overrides_truthy_and_falsey_values():
    cfg = reload_config(
        {
            "MAKER_PREFERENCE_OVERRIDES": (
                "BTC/USD:true,ETH/USD:no,SOL/USD:1,XRP/USD:0,ADA/USD:yes,LTC/USD:false,IGNORE"
            )
        }
    )

    assert cfg.MAKER_PREFERENCE_OVERRIDES == {
        "BTC/USD": True,
        "ETH/USD": False,
        "SOL/USD": True,
        "XRP/USD": False,
        "ADA/USD": True,
        "LTC/USD": False,
    }


def test_parse_correlation_buckets_ignores_empty_tokens_and_buckets():
    cfg = reload_config({"CORRELATION_BUCKETS": "BTC/USD,,ETH/USD; ;SOL/USD, ,ADA/USD;;"})

    assert cfg.CORRELATION_BUCKETS == {
        "bucket_1": ["BTC/USD", "ETH/USD"],
        "bucket_2": ["SOL/USD", "ADA/USD"],
    }


def test_client_order_prefix_defaults_when_version_missing():
    cfg = reload_config({})
    assert cfg.BOT_VERSION == "v1"
    assert cfg.CLIENT_ORDER_PREFIX == "BOT-v1"

    cfg_with_empty_version = reload_config({"BOT_VERSION": ""})
    assert cfg_with_empty_version.BOT_VERSION == "v1"
    assert cfg_with_empty_version.CLIENT_ORDER_PREFIX == "BOT-v1"

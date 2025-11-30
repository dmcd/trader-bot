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


def test_parse_maker_overrides_drops_invalid_entries():
    cfg = reload_config(
        {
            "MAKER_PREFERENCE_OVERRIDES": (
                "NO_COLON,BTC/USD:maybe,ETH/USD:,SOL/USD:  ,XRP/USD:true"
            )
        }
    )

    assert cfg.MAKER_PREFERENCE_OVERRIDES == {"XRP/USD": True}


def test_parse_correlation_buckets_with_malformed_groups():
    cfg = reload_config({"CORRELATION_BUCKETS": ";;BTC/USD,; ;ETH/USD;ADA/USD,,;SOL/USD,XRP/USD;;; "})

    assert cfg.CORRELATION_BUCKETS == {
        "bucket_1": ["BTC/USD"],
        "bucket_2": ["ETH/USD"],
        "bucket_3": ["ADA/USD"],
        "bucket_4": ["SOL/USD", "XRP/USD"],
    }


def test_ib_config_defaults_and_lists():
    cfg = reload_config({})

    assert cfg.IB_HOST == "127.0.0.1"
    assert cfg.IB_PORT == 7497
    assert cfg.IB_CLIENT_ID == 1
    assert cfg.IB_ACCOUNT_ID == ""
    assert cfg.IB_PAPER is True
    assert cfg.IB_BASE_CURRENCY == "AUD"
    assert cfg.IB_EXCHANGE == "SMART"
    assert cfg.IB_PRIMARY_EXCHANGE == "ASX"
    assert cfg.IB_ALLOWED_INSTRUMENT_TYPES == ["STK", "FX"]
    assert cfg.IB_STOCK_COMMISSION_PER_SHARE == 0.005
    assert cfg.IB_STOCK_MIN_COMMISSION == 1.0
    assert cfg.IB_FX_COMMISSION_PCT == 0.0


def test_ib_config_respects_overrides_and_uppercases_lists():
    cfg = reload_config(
        {
            "IB_HOST": "ib-paper",
            "IB_PORT": "4002",
            "IB_CLIENT_ID": "7",
            "IB_ACCOUNT_ID": "DU1234567",
            "IB_PAPER": "false",
            "IB_BASE_CURRENCY": "aud",
            "IB_EXCHANGE": "SMART",
            "IB_PRIMARY_EXCHANGE": "ASX",
            "IB_ALLOWED_INSTRUMENT_TYPES": "stk,fx , fut ",
            "IB_STOCK_COMMISSION_PER_SHARE": "0.003",
            "IB_STOCK_MIN_COMMISSION": "1.35",
            "IB_FX_COMMISSION_PCT": "0.00015",
        }
    )

    assert cfg.IB_HOST == "ib-paper"
    assert cfg.IB_PORT == 4002
    assert cfg.IB_CLIENT_ID == 7
    assert cfg.IB_ACCOUNT_ID == "DU1234567"
    assert cfg.IB_PAPER is False
    assert cfg.IB_BASE_CURRENCY == "AUD"
    assert cfg.IB_EXCHANGE == "SMART"
    assert cfg.IB_PRIMARY_EXCHANGE == "ASX"
    assert cfg.IB_ALLOWED_INSTRUMENT_TYPES == ["STK", "FX", "FUT"]
    assert cfg.IB_STOCK_COMMISSION_PER_SHARE == 0.003
    assert cfg.IB_STOCK_MIN_COMMISSION == 1.35
    assert cfg.IB_FX_COMMISSION_PCT == 0.00015

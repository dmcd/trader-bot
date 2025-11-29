import logging
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def fake_logger():
    """Shared lightweight logger mock for service unit tests."""
    logger = MagicMock(spec=logging.Logger)
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger

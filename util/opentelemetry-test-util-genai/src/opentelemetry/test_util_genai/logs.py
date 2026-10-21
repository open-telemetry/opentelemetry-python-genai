# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def assert_no_warnings(
    caplog: pytest.LogCaptureFixture, logger_name: str
) -> Iterator[None]:
    with caplog.at_level(logging.WARNING, logger=logger_name):
        yield
    leaked = [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno >= logging.WARNING
    ]
    assert not leaked, leaked

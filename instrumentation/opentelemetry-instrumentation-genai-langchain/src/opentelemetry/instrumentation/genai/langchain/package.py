# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


_instruments_langchain = "langchain >= 0.3.22, < 2"
_instruments_langchain_core = "langchain-core >= 0.3.49, < 2"
_instruments = ()
_instruments_any = (
    _instruments_langchain,
    _instruments_langchain_core,
)

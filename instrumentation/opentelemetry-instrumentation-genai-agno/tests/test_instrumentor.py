# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AgnoInstrumentor class."""

from __future__ import annotations

from opentelemetry.instrumentation.genai.agno import (
    AgnoInstrumentor,
)


def test_instrumentor_instantiation() -> None:
    """Test that the instrumentor can be instantiated."""
    instrumentor = AgnoInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, AgnoInstrumentor)


def test_instrumentation_dependencies() -> None:
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = AgnoInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "agno >= 2.0.0, < 3" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = AgnoInstrumentor()

    # First instrumentation
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # First uninstrumentation
    instrumentor.uninstrument()

    # Second instrumentation (should work)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second uninstrumentation
    instrumentor.uninstrument()


def test_multiple_instrumentation_calls(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that multiple instrument() calls don't cause issues."""
    instrumentor = AgnoInstrumentor()

    # First call
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second call (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_uninstrument_without_instrument() -> None:
    """Test that uninstrument() can be called without prior instrument()."""
    instrumentor = AgnoInstrumentor()

    # This should not raise an error
    instrumentor.uninstrument()


def test_instrument_with_no_providers() -> None:
    """Test that instrument() works without explicit providers."""
    instrumentor = AgnoInstrumentor()

    # Should use global providers
    instrumentor.instrument()

    # Clean up
    instrumentor.uninstrument()


def test_instrumentor_has_required_attributes() -> None:
    """Test that the instrumentor has the required methods."""
    instrumentor = AgnoInstrumentor()

    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "instrumentation_dependencies")
    assert callable(instrumentor.instrument)
    assert callable(instrumentor.uninstrument)
    assert callable(instrumentor.instrumentation_dependencies)


def test_deferred_wrapping_disabled_after_uninstrument(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that deferred post-import hooks do not patch modules imported after uninstrument."""
    import sys
    import types

    from wrapt import notify_module_loaded

    from opentelemetry.instrumentation.genai.agno import patch as patch_module

    fake_mod_name = "agno.test_deferred_uninstrument_module"
    if fake_mod_name in sys.modules:
        del sys.modules[fake_mod_name]

    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    called: list[bool] = []

    def dummy_wrapper(wrapped, instance, args, kwargs):
        called.append(True)
        return wrapped(*args, **kwargs)

    patch_module._safe_wrap_function(
        fake_mod_name,
        "TargetClass.action",
        dummy_wrapper,
        patch_module._instrumentation_generation,
    )

    # Uninstrument before module is loaded
    instrumentor.uninstrument()

    # Now module is imported
    fake_mod = types.ModuleType(fake_mod_name)

    class TargetClass:
        def action(self) -> str:
            return "original"

    fake_mod.TargetClass = TargetClass
    sys.modules[fake_mod_name] = fake_mod

    notify_module_loaded(fake_mod)

    try:
        assert not hasattr(TargetClass.action, "__wrapped__")
        assert TargetClass().action() == "original"
        assert len(called) == 0
    finally:
        del sys.modules[fake_mod_name]


def test_deferred_wrapping_re_instrument(
    tracer_provider, logger_provider, meter_provider
) -> None:
    """Test that re-instrumenting patches correctly without stacking wrappers."""
    import sys
    import types

    from wrapt import notify_module_loaded

    from opentelemetry.instrumentation.genai.agno import patch as patch_module

    fake_mod_name = "agno.test_deferred_reinstrument_module"
    if fake_mod_name in sys.modules:
        del sys.modules[fake_mod_name]

    instrumentor = AgnoInstrumentor()
    # Cycle 1: instrument then uninstrument
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()

    # Cycle 2: re-instrument
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    call_count = 0

    def dummy_wrapper(wrapped, instance, args, kwargs):
        nonlocal call_count
        call_count += 1
        return wrapped(*args, **kwargs)

    patch_module._safe_wrap_function(
        fake_mod_name,
        "TargetClass.action",
        dummy_wrapper,
        patch_module._instrumentation_generation,
    )

    fake_mod = types.ModuleType(fake_mod_name)

    class TargetClass:
        def action(self) -> str:
            return "original"

    fake_mod.TargetClass = TargetClass
    sys.modules[fake_mod_name] = fake_mod

    notify_module_loaded(fake_mod)

    try:
        assert hasattr(TargetClass.action, "__wrapped__")
        assert TargetClass().action() == "original"
        assert call_count == 1
    finally:
        instrumentor.uninstrument()
        if fake_mod_name in sys.modules:
            del sys.modules[fake_mod_name]

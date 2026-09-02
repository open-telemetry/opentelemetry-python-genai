# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# tests/test_invocation_manager.py
import uuid
from unittest import mock

import pytest

from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
    _InvocationManager,
)
from opentelemetry.util.genai.types import GenAIInvocation


@pytest.fixture
def invocation_manager():
    return _InvocationManager()


@pytest.fixture
def mock_invocation():
    return mock.Mock(spec=GenAIInvocation)


def test_add_invocation_state_without_parent(
    invocation_manager, mock_invocation
):
    run_id = uuid.uuid4()
    invocation_manager.add_invocation_state(
        run_id=run_id,
        parent_run_id=None,
        invocation=mock_invocation,
    )

    assert invocation_manager.get_invocation(run_id) == mock_invocation
    assert len(invocation_manager._invocations) == 1
    assert invocation_manager._invocations[run_id].children == []


def test_add_invocation_state_with_parent(invocation_manager):
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()
    parent_invocation = mock.Mock(spec=GenAIInvocation)
    child_invocation = mock.Mock(spec=GenAIInvocation)

    # Add parent first
    invocation_manager.add_invocation_state(
        run_id=parent_id,
        parent_run_id=None,
        invocation=parent_invocation,
    )

    # Then add child with parent reference
    invocation_manager.add_invocation_state(
        run_id=child_id,
        parent_run_id=parent_id,
        invocation=child_invocation,
    )

    # Check that parent has child in its children list
    assert child_id in invocation_manager._invocations[parent_id].children
    assert invocation_manager.get_invocation(child_id) == child_invocation
    assert invocation_manager.get_invocation(parent_id) == parent_invocation


def test_add_invocation_state_with_nonexistent_parent(
    invocation_manager, mock_invocation
):
    run_id = uuid.uuid4()
    nonexistent_parent_id = uuid.uuid4()

    # Adding with a parent that doesn't exist should still add the child without error
    invocation_manager.add_invocation_state(
        run_id=run_id,
        parent_run_id=nonexistent_parent_id,
        invocation=mock_invocation,
    )

    assert invocation_manager.get_invocation(run_id) == mock_invocation
    assert len(invocation_manager._invocations) == 1


def test_get_nonexistent_invocation(invocation_manager):
    nonexistent_id = uuid.uuid4()
    assert invocation_manager.get_invocation(nonexistent_id) is None


def test_delete_invocation_state(invocation_manager, mock_invocation):
    run_id = uuid.uuid4()
    invocation_manager.add_invocation_state(
        run_id=run_id,
        parent_run_id=None,
        invocation=mock_invocation,
    )

    # Verify it was added
    assert invocation_manager.get_invocation(run_id) == mock_invocation

    # Delete it
    invocation_manager.delete_invocation_state(run_id)

    # Verify it was removed
    assert run_id not in invocation_manager._invocations


def test_delete_invocation_state_deferred_while_children_live(
    invocation_manager,
):
    """Deleting a parent while children are still live defers its removal."""
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()

    parent_invocation = mock.Mock(spec=GenAIInvocation)
    child_invocation = mock.Mock(spec=GenAIInvocation)

    invocation_manager.add_invocation_state(
        run_id=parent_id, parent_run_id=None, invocation=parent_invocation
    )
    invocation_manager.add_invocation_state(
        run_id=child_id, parent_run_id=parent_id, invocation=child_invocation
    )

    # Delete the parent while the child is still live
    invocation_manager.delete_invocation_state(parent_id)

    # Parent should still be present (deferred) because child is live
    assert parent_id in invocation_manager._invocations
    assert invocation_manager._invocations[parent_id].ended is True

    # After the child is deleted, the parent should also be cleaned up
    invocation_manager.delete_invocation_state(child_id)

    assert child_id not in invocation_manager._invocations
    assert parent_id not in invocation_manager._invocations


def test_delete_invocation_state_propagates_upward(invocation_manager):
    """When the last child is removed, an already-ended parent is cleaned up."""
    grandparent_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()

    for run_id, parent in [
        (grandparent_id, None),
        (parent_id, grandparent_id),
        (child_id, parent_id),
    ]:
        invocation_manager.add_invocation_state(
            run_id=run_id,
            parent_run_id=parent,
            invocation=mock.Mock(spec=GenAIInvocation),
        )

    # Mark grandparent and parent as ended (deferred)
    invocation_manager.delete_invocation_state(grandparent_id)
    invocation_manager.delete_invocation_state(parent_id)

    assert grandparent_id in invocation_manager._invocations  # deferred
    assert parent_id in invocation_manager._invocations  # deferred

    # Removing the last live node should cascade upward
    invocation_manager.delete_invocation_state(child_id)

    assert child_id not in invocation_manager._invocations
    assert parent_id not in invocation_manager._invocations
    assert grandparent_id not in invocation_manager._invocations


def test_delete_invocation_state_with_multiple_children_defers_until_last(
    invocation_manager,
):
    """Parent removal is deferred until all children are gone."""
    parent_id = uuid.uuid4()
    child1_id = uuid.uuid4()
    child2_id = uuid.uuid4()

    parent_invocation = mock.Mock(spec=GenAIInvocation)
    child1_invocation = mock.Mock(spec=GenAIInvocation)
    child2_invocation = mock.Mock(spec=GenAIInvocation)

    invocation_manager.add_invocation_state(
        run_id=parent_id, parent_run_id=None, invocation=parent_invocation
    )
    invocation_manager.add_invocation_state(
        run_id=child1_id, parent_run_id=parent_id, invocation=child1_invocation
    )
    invocation_manager.add_invocation_state(
        run_id=child2_id, parent_run_id=parent_id, invocation=child2_invocation
    )

    # Delete parent while both children live → deferred
    invocation_manager.delete_invocation_state(parent_id)
    assert parent_id in invocation_manager._invocations

    # Remove first child → parent still deferred (child2 is live)
    invocation_manager.delete_invocation_state(child1_id)
    assert parent_id in invocation_manager._invocations

    # Remove last child → parent is now cleaned up
    invocation_manager.delete_invocation_state(child2_id)
    assert parent_id not in invocation_manager._invocations


def test_get_parent_run_id_returns_none_for_unknown(invocation_manager):
    assert invocation_manager.get_parent_run_id(uuid.uuid4()) is None


def test_get_parent_run_id_returns_registered_parent(invocation_manager):
    parent_id = uuid.uuid4()
    child_id = uuid.uuid4()

    invocation_manager.add_invocation_state(
        run_id=parent_id,
        parent_run_id=None,
        invocation=mock.Mock(spec=GenAIInvocation),
    )
    invocation_manager.add_invocation_state(
        run_id=child_id,
        parent_run_id=parent_id,
        invocation=mock.Mock(spec=GenAIInvocation),
    )

    assert invocation_manager.get_parent_run_id(child_id) == parent_id
    assert invocation_manager.get_parent_run_id(parent_id) is None


def test_none_invocation_can_be_stored_and_retrieved(invocation_manager):
    """Nodes with no associated span (None invocation) must still be tracked."""
    run_id = uuid.uuid4()

    invocation_manager.add_invocation_state(
        run_id=run_id, parent_run_id=None, invocation=None
    )

    assert run_id in invocation_manager._invocations
    assert invocation_manager.get_invocation(run_id) is None


def test_thread_binding_resolves_the_running_invocation(
    invocation_manager, mock_invocation
):
    run_id = uuid.uuid4()
    invocation_manager.add_invocation_state(
        run_id=run_id, parent_run_id=None, invocation=mock_invocation
    )

    invocation_manager.bind_thread("thread-1", run_id)

    assert (
        invocation_manager.get_thread_invocation("thread-1") is mock_invocation
    )
    assert invocation_manager.get_thread_invocation("thread-2") is None

    invocation_manager.unbind_thread(run_id)

    assert invocation_manager.get_thread_invocation("thread-1") is None


def test_delete_nonexistent_run_id_does_not_raise(invocation_manager):
    invocation_manager.delete_invocation_state(uuid.uuid4())  # must not raise


# ---------------------------------------------------------------------------
# LangGraph thread correlation
# ---------------------------------------------------------------------------


def _bind_run(manager, thread_id, parent_run_id=None):
    run_id = uuid.uuid4()
    invocation = mock.Mock(spec=GenAIInvocation)
    manager.add_invocation_state(
        run_id=run_id, parent_run_id=parent_run_id, invocation=invocation
    )
    manager.bind_thread(thread_id, run_id)
    return run_id, invocation


def test_nested_run_never_displaces_the_run_containing_it(invocation_manager):
    outer_run_id, outer_invocation = _bind_run(invocation_manager, "t")
    _, inner_invocation = _bind_run(
        invocation_manager, "t", parent_run_id=outer_run_id
    )

    # The innermost live run owns the checkpoint.
    assert invocation_manager.get_thread_invocation("t") is inner_invocation

    # When the nested run ends, the run containing it keeps the thread id.
    inner_run_id = invocation_manager._threads["t"][-1][1]
    invocation_manager.unbind_thread(inner_run_id)
    invocation_manager.delete_invocation_state(inner_run_id)

    assert invocation_manager._threads["t"] == [("", outer_run_id)]
    assert invocation_manager.get_thread_invocation("t") is outer_invocation


def test_unrelated_concurrent_runs_on_one_thread_are_dropped(
    invocation_manager,
):
    _bind_run(invocation_manager, "t")
    _bind_run(invocation_manager, "t")

    # Ownership is ambiguous, so nothing is returned rather than guessed.
    assert invocation_manager.get_thread_invocation("t") is None


def test_dead_runs_are_pruned_from_the_thread_binding(invocation_manager):
    abandoned_run_id, _ = _bind_run(invocation_manager, "t")
    invocation_manager.delete_invocation_state(abandoned_run_id)

    _, live_invocation = _bind_run(invocation_manager, "t")

    assert len(invocation_manager._threads["t"]) == 1
    assert invocation_manager.get_thread_invocation("t") is live_invocation


def test_thread_bindings_are_bounded_for_runs_that_never_end(
    invocation_manager,
):
    from opentelemetry.instrumentation.genai.langchain.invocation_manager import (
        _MAX_RUNS_PER_THREAD,
        _MAX_TRACKED_THREADS,
    )

    for _ in range(_MAX_RUNS_PER_THREAD + 10):
        _bind_run(invocation_manager, "abandoned-thread")
    assert (
        len(invocation_manager._threads["abandoned-thread"])
        == _MAX_RUNS_PER_THREAD
    )

    for index in range(_MAX_TRACKED_THREADS + 5):
        _bind_run(invocation_manager, f"thread-{index}")
    assert len(invocation_manager._threads) <= _MAX_TRACKED_THREADS


def test_thread_lookup_is_atomic_against_concurrent_run_teardown(
    invocation_manager,
):
    import threading

    results = []
    errors = []

    def churn():
        try:
            for _ in range(200):
                run_id, invocation = _bind_run(invocation_manager, "t")
                invocation_manager.unbind_thread(run_id)
                invocation_manager.delete_invocation_state(run_id)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def read():
        try:
            for _ in range(200):
                results.append(invocation_manager.get_thread_invocation("t"))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=churn), threading.Thread(target=read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    # Every answer is either a live invocation or nothing, never a torn read.
    assert all(
        result is None or isinstance(result, mock.Mock) for result in results
    )


def test_checkpoint_namespace_selects_the_run_that_owns_it(invocation_manager):
    root_run_id = uuid.uuid4()
    child_run_id = uuid.uuid4()
    root_invocation = mock.Mock(spec=GenAIInvocation)
    child_invocation = mock.Mock(spec=GenAIInvocation)
    invocation_manager.add_invocation_state(
        run_id=root_run_id, parent_run_id=None, invocation=root_invocation
    )
    invocation_manager.add_invocation_state(
        run_id=child_run_id,
        parent_run_id=root_run_id,
        invocation=child_invocation,
    )
    invocation_manager.bind_thread("t", root_run_id, "")
    invocation_manager.bind_thread("t", child_run_id, "child:abc")

    # A write made under the child namespace belongs to the child run.
    assert (
        invocation_manager.get_thread_invocation("t", "child:abc")
        is child_invocation
    )
    # And so does a write from a graph nested inside the child.
    assert (
        invocation_manager.get_thread_invocation("t", "child:abc|leaf:def")
        is child_invocation
    )
    # A root namespace write belongs to the root run.
    assert invocation_manager.get_thread_invocation("t", "") is root_invocation
    # A namespace no run is bound for falls back to the nearest enclosing run.
    assert (
        invocation_manager.get_thread_invocation("t", "other:xyz")
        is root_invocation
    )


def test_namespace_write_falls_back_when_the_child_run_has_no_binding(
    invocation_manager,
):
    root_run_id, root_invocation = _bind_run(invocation_manager, "t")

    # Nothing is bound for the child namespace, so the enclosing run owns it.
    assert (
        invocation_manager.get_thread_invocation("t", "child:abc")
        is root_invocation
    )
    assert root_run_id is not None

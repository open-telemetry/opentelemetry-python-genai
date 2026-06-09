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


def test_delete_nonexistent_run_id_does_not_raise(invocation_manager):
    invocation_manager.delete_invocation_state(uuid.uuid4())  # must not raise

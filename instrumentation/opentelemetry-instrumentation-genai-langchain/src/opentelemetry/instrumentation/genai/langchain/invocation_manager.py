# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import TypeAlias
from uuid import UUID

from opentelemetry.context import Context
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
    LocalAgentInvocation,
    RetrievalInvocation,
    ToolInvocation,
    WorkflowInvocation,
)

__all__ = ["_InvocationManager"]

_AnyInvocation: TypeAlias = (
    InferenceInvocation
    | ToolInvocation
    | WorkflowInvocation
    | LocalAgentInvocation
    | RetrievalInvocation
)


@dataclass
class _InvocationState:
    invocation: _AnyInvocation | None
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False
    agent_name: str | None = None


class _InvocationManager:
    def __init__(self) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}

    def add_invocation_state(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        invocation: _AnyInvocation | None,
        agent_name: str | None = None,
    ) -> None:
        invocation_state = _InvocationState(
            invocation=invocation,
            agent_name=agent_name,
        )

        invocation_state.parent_run_id = parent_run_id
        if parent_run_id is not None and parent_run_id in self._invocations:
            parent_invocation_state = self._invocations[parent_run_id]
            parent_invocation_state.children.append(run_id)

        self._invocations[run_id] = invocation_state

    def get_invocation(self, run_id: UUID) -> _AnyInvocation | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.invocation if invocation_state else None

    def get_agent_name(self, run_id: UUID) -> str | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.agent_name if invocation_state else None

    def get_parent_run_id(self, run_id: UUID) -> UUID | None:
        invocation_state = self._invocations.get(run_id)
        return invocation_state.parent_run_id if invocation_state else None

    def get_parent_context(self, parent_run_id: UUID | None) -> Context | None:
        current = parent_run_id
        while current is not None:
            invocation = self.get_invocation(current)
            if invocation is not None:
                return invocation.context
            current = self.get_parent_run_id(current)
        return None

    def delete_invocation_state(self, run_id: UUID) -> None:
        invocation_state = self._invocations.get(run_id)
        if not invocation_state:
            return

        invocation_state.ended = True

        # Defer removal if any children are still live, so upward traversal
        # via _find_agent_context can still walk through this node.
        if any(c in self._invocations for c in invocation_state.children):
            return

        self._invocations.pop(run_id, None)

        # Propagate cleanup upward: if the parent has already ended and has no
        # more live children, it can now be removed too.
        if invocation_state.parent_run_id:
            parent_state = self._invocations.get(
                invocation_state.parent_run_id
            )
            if parent_state is not None and parent_state.ended:
                self.delete_invocation_state(invocation_state.parent_run_id)

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
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

__all__ = ["_InvocationManager", "_PromptContext"]

_AnyInvocation: TypeAlias = (
    InferenceInvocation
    | ToolInvocation
    | WorkflowInvocation
    | LocalAgentInvocation
    | RetrievalInvocation
)


@dataclass(frozen=True)
class _PromptContext:
    name: str | None
    variables: Mapping[str, object]


@dataclass
class _InferenceCallScope:
    run_id: UUID
    parent: _InferenceCallScope | None = None
    positions: dict[UUID, int] = field(
        default_factory=lambda: dict[UUID, int]()
    )
    keys: dict[UUID, list[tuple[UUID, int]]] = field(
        default_factory=lambda: dict[UUID, list[tuple[UUID, int]]]()
    )

    def key(self, agent_run_id: UUID) -> tuple[UUID, int]:
        position = self.positions.get(agent_run_id, 0)
        self.positions[agent_run_id] = position + 1
        keys = self.keys.setdefault(agent_run_id, [])
        if position == len(keys):
            # Repeated inner attempts reuse their key without advancing the parent.
            keys.append(
                self.parent.key(agent_run_id)
                if self.parent is not None
                else (self.run_id, position)
            )
        return keys[position]

    def is_nested_in(self, scope: _InferenceCallScope) -> bool:
        current: _InferenceCallScope | None = self
        while current is not None:
            if current is scope:
                return True
            current = current.parent
        return False

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _inference_calls.set(self)
        try:
            yield
        finally:
            _inference_calls.reset(token)


_inference_calls: ContextVar[_InferenceCallScope | None] = ContextVar(
    "otel_langchain_inference_calls", default=None
)
_INFERENCE_CALLS_KEY = "__otel_langchain_inference_calls"


@dataclass
class _InvocationState:
    invocation: _AnyInvocation | None
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False
    agent_name: str | None = None
    inference_scope: _InferenceCallScope | None = None
    inference_call_keys: set[tuple[UUID, int]] = field(
        default_factory=lambda: set[tuple[UUID, int]]()
    )
    prompt_context: _PromptContext | None = None
    pending_prompt_contexts: list[_PromptContext] = field(
        default_factory=lambda: list[_PromptContext]()
    )


class _InvocationManager:
    def __init__(self) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}
        self._call_count_lock = Lock()

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

    def get_inference_scope(
        self,
        parent_run_id: UUID | None,
        configured: _InferenceCallScope | None = None,
    ) -> _InferenceCallScope | None:
        scope = _inference_calls.get()
        if configured is not None and (
            scope is None or configured.is_nested_in(scope)
        ):
            scope = configured
        current = parent_run_id
        while current is not None:
            state = self._invocations.get(current)
            if state is None:
                break
            if state.inference_scope is not None:
                candidate = state.inference_scope
                if scope is None or candidate.is_nested_in(scope):
                    scope = candidate
                break
            if isinstance(
                state.invocation,
                (
                    LocalAgentInvocation,
                    InferenceInvocation,
                    ToolInvocation,
                    RetrievalInvocation,
                ),
            ):
                break
            current = state.parent_run_id
        return scope

    def start_inference_attempt(
        self,
        run_id: UUID,
        configured: _InferenceCallScope | None = None,
    ) -> _InferenceCallScope | None:
        with self._call_count_lock:
            state = self._invocations.get(run_id)
            if state is None:
                return None
            if state.inference_scope is None:
                state.inference_scope = _InferenceCallScope(
                    run_id,
                    self.get_inference_scope(state.parent_run_id, configured),
                )
            else:
                scope = state.inference_scope
                state.inference_scope = _InferenceCallScope(
                    run_id, scope.parent, keys=scope.keys
                )
            return state.inference_scope

    def record_agent_call(
        self, parent_run_id: UUID | None, *, tool_call: bool = False
    ) -> None:
        with self._call_count_lock:
            scope = (
                None if tool_call else self.get_inference_scope(parent_run_id)
            )
            current = parent_run_id
            while current is not None:
                state = self._invocations.get(current)
                invocation = self.get_invocation(current)
                if isinstance(invocation, LocalAgentInvocation):
                    if tool_call:
                        if invocation.tool_calls is not None:
                            invocation.tool_calls += 1
                    elif invocation.inference_calls is not None:
                        if scope is not None and state is not None:
                            key = scope.key(current)
                            if key in state.inference_call_keys:
                                return
                            state.inference_call_keys.add(key)
                        invocation.inference_calls += 1
                    return
                if isinstance(
                    invocation,
                    (InferenceInvocation, ToolInvocation, RetrievalInvocation),
                ):
                    return
                current = self.get_parent_run_id(current)

    def set_prompt_context(
        self, run_id: UUID, prompt_context: _PromptContext
    ) -> None:
        invocation_state = self._invocations.get(run_id)
        if invocation_state is not None:
            invocation_state.prompt_context = prompt_context

    def publish_prompt_context(self, run_id: UUID) -> None:
        invocation_state = self._invocations.get(run_id)
        if invocation_state is None or invocation_state.prompt_context is None:
            return
        parent_run_id = invocation_state.parent_run_id
        parent_state = (
            self._invocations.get(parent_run_id)
            if parent_run_id is not None
            else None
        )
        if parent_state is not None:
            parent_state.pending_prompt_contexts.append(
                invocation_state.prompt_context
            )
        invocation_state.prompt_context = None

    def consume_prompt_context(
        self, parent_run_id: UUID | None
    ) -> _PromptContext | None:
        current_run_id = parent_run_id
        while current_run_id is not None:
            invocation_state = self._invocations.get(current_run_id)
            if invocation_state is None:
                return None
            if invocation_state.pending_prompt_contexts:
                prompt_context = invocation_state.pending_prompt_contexts[-1]
                invocation_state.pending_prompt_contexts.clear()
                return prompt_context
            current_run_id = invocation_state.parent_run_id
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

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import weakref
from collections.abc import Mapping
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
    version: str | None = None


@dataclass
class _InvocationState:
    invocation: _AnyInvocation | None
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False
    agent_name: str | None = None
    prompt_context: _PromptContext | None = None
    pending_prompt_contexts: list[_PromptContext] = field(
        default_factory=lambda: list[_PromptContext]()
    )


class _InvocationManager:
    def __init__(self) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}
        # Prompt contexts of the prompt values that prompt runs returned, keyed
        # by object id while the value is alive.
        self._prompt_values: dict[int, _PromptContext] = {}
        # Prompt contexts of chat model runs whose input is one of those values,
        # keyed by the run id chosen before the run starts.
        self._model_prompt_contexts: dict[UUID, _PromptContext] = {}

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

    def set_prompt_context(
        self, run_id: UUID, prompt_context: _PromptContext
    ) -> None:
        invocation_state = self._invocations.get(run_id)
        if invocation_state is not None:
            invocation_state.prompt_context = prompt_context

    def publish_prompt_context(self, run_id: UUID) -> _PromptContext | None:
        invocation_state = self._invocations.get(run_id)
        if invocation_state is None or invocation_state.prompt_context is None:
            return None
        prompt_context = invocation_state.prompt_context
        parent_run_id = invocation_state.parent_run_id
        parent_state = (
            self._invocations.get(parent_run_id)
            if parent_run_id is not None
            else None
        )
        if parent_state is not None:
            parent_state.pending_prompt_contexts.append(prompt_context)
        invocation_state.prompt_context = None
        return prompt_context

    def set_prompt_value_context(
        self, prompt_value: object, prompt_context: _PromptContext
    ) -> None:
        """Remember which template produced ``prompt_value``, while it is alive."""
        key = id(prompt_value)
        try:
            weakref.finalize(prompt_value, self._prompt_values.pop, key, None)
        except TypeError:
            return  # not weak-referenceable, so its id could be reused
        self._prompt_values[key] = prompt_context

    def get_prompt_value_context(
        self, prompt_value: object
    ) -> _PromptContext | None:
        return self._prompt_values.get(id(prompt_value))

    def set_model_prompt_context(
        self, run_id: UUID, prompt_context: _PromptContext
    ) -> None:
        self._model_prompt_contexts[run_id] = prompt_context

    def pop_model_prompt_context(self, run_id: UUID) -> _PromptContext | None:
        return self._model_prompt_contexts.pop(run_id, None)

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

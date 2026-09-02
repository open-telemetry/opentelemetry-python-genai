# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from dataclasses import dataclass, field
from uuid import UUID

from opentelemetry.util.genai.types import GenAIInvocation

__all__ = ["_InvocationManager"]

_logger = logging.getLogger(__name__)

# Upper bound on LangGraph thread ids tracked at once, and on runs tracked per
# thread id. Runs that never report an end (a crashed worker, a cancelled task)
# would otherwise accumulate forever.
_MAX_TRACKED_THREADS = 512
_MAX_RUNS_PER_THREAD = 32

# LangGraph joins nested checkpoint namespaces with this separator
# (``langgraph.constants.NS_SEP``).
_NAMESPACE_SEPARATOR = "|"


def _namespace_contains(outer: str, inner: str) -> bool:
    """Return whether ``outer`` is ``inner`` or encloses it."""
    if not outer:
        return True
    return inner == outer or inner.startswith(outer + _NAMESPACE_SEPARATOR)


@dataclass
class _InvocationState:
    invocation: GenAIInvocation | None
    children: list[UUID] = field(default_factory=lambda: list())
    parent_run_id: UUID | None = None
    ended: bool = False


class _InvocationManager:
    def __init__(
        self,
    ) -> None:
        # Map from run_id -> _InvocationState, to keep track of invocations and parent/child relationships
        # TODO: TTL cache to avoid memory leaks in long-running processes.
        self._invocations: dict[UUID, _InvocationState] = {}
        # Map from LangGraph thread id -> the live graph runs on that thread
        # id, outermost first, each paired with the checkpoint namespace its
        # own checkpoints are written under ("" for a top level graph). A
        # checkpointer call carries no callback run id, so the thread id and
        # namespace in its config are the only handle back to a running
        # invocation. This is a list rather than a single id so that a nested
        # graph run never displaces the run that contains it.
        self._threads: dict[str, list[tuple[str, UUID]]] = {}
        # One lock guards both maps: a checkpoint lookup reads the thread stack
        # and the invocation states together, and must not observe a run being
        # torn down halfway through by a concurrent chain end.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # LangGraph thread correlation
    # ------------------------------------------------------------------

    def bind_thread(
        self, thread_id: str, run_id: UUID, namespace: str = ""
    ) -> None:
        """Record that ``run_id`` is a live graph run on ``thread_id``.

        ``namespace`` is the LangGraph checkpoint namespace the run's own
        checkpoints are written under: "" for a top level graph, and the
        run's ``langgraph_checkpoint_ns`` for a nested graph.
        """
        with self._lock:
            runs = self._threads.setdefault(thread_id, [])
            # Drop runs whose invocation state is already gone, so abandoned
            # runs do not pin an entry forever.
            runs[:] = [
                entry for entry in runs if entry[1] in self._invocations
            ]
            if all(entry[1] != run_id for entry in runs):
                runs.append((namespace, run_id))
            del runs[:-_MAX_RUNS_PER_THREAD]
            self._prune_threads()

    def unbind_thread(self, run_id: UUID) -> None:
        """Forget ``run_id``, keeping any run that contains or follows it."""
        with self._lock:
            for thread_id, runs in list(self._threads.items()):
                runs[:] = [entry for entry in runs if entry[1] != run_id]
                if not runs:
                    del self._threads[thread_id]

    def get_thread_invocation(
        self, thread_id: str, namespace: str = ""
    ) -> GenAIInvocation | None:
        """Return the invocation a checkpoint write belongs to.

        A write carries the namespace of the graph that produced it. The run
        bound with that exact namespace owns it; if no run is bound there, the
        nearest enclosing run does, which is the live run whose namespace is
        the longest prefix of the write's namespace. If two live runs are
        equally specific and unrelated, ownership is genuinely ambiguous and
        nothing is returned: a miscorrelated event is worse than a missing one.
        """
        with self._lock:
            candidates = [
                entry
                for entry in self._threads.get(thread_id, [])
                if entry[1] in self._invocations
                and _namespace_contains(entry[0], namespace)
            ]
            if not candidates:
                return None

            longest = max(len(entry[0]) for entry in candidates)
            matches = [
                entry for entry in candidates if len(entry[0]) == longest
            ]
            if len(matches) > 1:
                # Equally specific runs are only unambiguous when they nest.
                innermost = matches[-1][1]
                ancestors = self._ancestors(innermost)
                if not all(entry[1] in ancestors for entry in matches[:-1]):
                    _logger.debug(
                        "Ambiguous LangGraph thread id %s namespace %r: %d "
                        "unrelated live runs, dropping the checkpoint event",
                        thread_id,
                        namespace,
                        len(matches),
                    )
                    return None

            state = self._invocations.get(matches[-1][1])
            return state.invocation if state else None

    def _ancestors(self, run_id: UUID) -> set[UUID]:
        """Return the run ids between ``run_id`` and the root, exclusive."""
        ancestors: set[UUID] = set()
        state = self._invocations.get(run_id)
        while state is not None and state.parent_run_id is not None:
            parent_run_id = state.parent_run_id
            if parent_run_id in ancestors:
                break
            ancestors.add(parent_run_id)
            state = self._invocations.get(parent_run_id)
        return ancestors

    def _prune_threads(self) -> None:
        """Drop the oldest thread ids once the tracked set grows too large."""
        overflow = len(self._threads) - _MAX_TRACKED_THREADS
        if overflow <= 0:
            return
        for thread_id in list(self._threads)[:overflow]:
            del self._threads[thread_id]

    # ------------------------------------------------------------------
    # Run state
    # ------------------------------------------------------------------

    def add_invocation_state(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        invocation: GenAIInvocation | None,
    ) -> None:
        invocation_state = _InvocationState(invocation=invocation)

        with self._lock:
            if (
                parent_run_id is not None
                and parent_run_id in self._invocations
            ):
                invocation_state.parent_run_id = parent_run_id

                parent_invocation_state = self._invocations[parent_run_id]
                parent_invocation_state.children.append(run_id)

            self._invocations[run_id] = invocation_state

    def get_invocation(self, run_id: UUID) -> GenAIInvocation | None:
        with self._lock:
            invocation_state = self._invocations.get(run_id)
            return invocation_state.invocation if invocation_state else None

    def get_parent_run_id(self, run_id: UUID) -> UUID | None:
        with self._lock:
            invocation_state = self._invocations.get(run_id)
            return invocation_state.parent_run_id if invocation_state else None

    def delete_invocation_state(self, run_id: UUID) -> None:
        with self._lock:
            invocation_state = self._invocations.get(run_id)
            if not invocation_state:
                return

            invocation_state.ended = True

            # Defer removal if any children are still live, so upward traversal
            # (e.g. _find_nearest_agent) can still walk through this node.
            if any(c in self._invocations for c in invocation_state.children):
                return

            self._invocations.pop(run_id, None)

            # Propagate cleanup upward: if the parent has already ended and has
            # no more live children, it can now be removed too.
            if invocation_state.parent_run_id:
                parent_state = self._invocations.get(
                    invocation_state.parent_run_id
                )
                if parent_state is not None and parent_state.ended:
                    self.delete_invocation_state(
                        invocation_state.parent_run_id
                    )

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent lifecycle telemetry captured from unmodified LangGraph applications.

Every id asserted here comes from LangGraph itself: interrupt ids from the
``Interrupt`` objects the graph returns, checkpoint ids from the checkpointer's
own return values.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, TypedDict

import pytest

from opentelemetry.instrumentation.genai.langchain.lifecycle import (
    _wrap_checkpointer,
)

InMemorySaver = pytest.importorskip(
    "langgraph.checkpoint.memory"
).InMemorySaver
pytest.importorskip("langgraph.callbacks")
_graph = pytest.importorskip("langgraph.graph")
_types = pytest.importorskip("langgraph.types")
END = _graph.END
START = _graph.START
StateGraph = _graph.StateGraph
Command = _types.Command
interrupt = _types.interrupt

PAUSED = "gen_ai.agent.paused"
CHECKPOINTED = "gen_ai.agent.checkpointed"
RESUMED = "gen_ai.agent.resumed"
LIFECYCLE_EVENTS = (PAUSED, CHECKPOINTED, RESUMED)


class _State(TypedDict, total=False):
    value: str
    approval: str


def _build_graph():
    def start(_state: _State) -> _State:
        return {"value": "prepared"}

    def approval(_state: _State) -> _State:
        return {"approval": str(interrupt({"question": "approve?"}))}

    def finish(_state: _State) -> _State:
        return {"value": "submitted"}

    builder = StateGraph(_State)
    builder.add_node("start", start)
    builder.add_node("approval", approval)
    builder.add_node("finish", finish)
    builder.add_edge(START, "start")
    builder.add_edge("start", "approval")
    builder.add_edge("approval", "finish")
    builder.add_edge("finish", END)
    return builder


def _events(log_exporter, event_name: str) -> list[Any]:
    return [
        item.log_record
        for item in log_exporter.get_finished_logs()
        if item.log_record.event_name == event_name
    ]


def _workflow_spans(span_exporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]


def test_interrupt_and_resume_emit_correlated_lifecycle_events(
    start_instrumentation,
    log_exporter,
    span_exporter,
):
    graph = _build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "lifecycle-test"}}

    paused_result = graph.invoke({"value": "new"}, config=config)
    interrupts = paused_result["__interrupt__"]
    assert len(interrupts) == 1
    real_interrupt_id = interrupts[0].id

    # paused carries the id LangGraph minted for this interrupt.
    paused = _events(log_exporter, PAUSED)
    assert len(paused) == 1
    assert paused[0].attributes["gen_ai.agent.pause.id"] == real_interrupt_id
    pause_checkpoint = paused[0].attributes["gen_ai.agent.checkpoint.id"]
    # No pause reason: LangGraph does not report why execution paused.
    assert "gen_ai.agent.pause.reason" not in paused[0].attributes
    # No execution id: LangGraph has no id spanning suspend and resume.
    assert "gen_ai.agent.execution.id" not in paused[0].attributes

    # LangGraph writes one checkpoint per superstep, so checkpointed is a
    # per-step record: the input checkpoint plus one per completed superstep.
    first_run_checkpoints = [
        record.attributes["gen_ai.agent.checkpoint.id"]
        for record in _events(log_exporter, CHECKPOINTED)
    ]
    assert len(first_run_checkpoints) == 3
    assert len(set(first_run_checkpoints)) == 3
    # The graph pauses at the checkpoint it last persisted.
    assert pause_checkpoint == first_run_checkpoints[-1]

    resumed_result = graph.invoke(Command(resume="approved"), config=config)
    assert resumed_result["approval"] == "approved"

    resumed = _events(log_exporter, RESUMED)
    assert len(resumed) == 1
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.type"] == "checkpoint"
    )
    # The resume continues from exactly the checkpoint the pause reported.
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.id"]
        == pause_checkpoint
    )

    second_run_checkpoints = [
        record.attributes["gen_ai.agent.checkpoint.id"]
        for record in _events(log_exporter, CHECKPOINTED)
    ][len(first_run_checkpoints) :]
    assert len(second_run_checkpoints) == 2

    # Each event is correlated with the workflow span of its own invoke call.
    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2
    paused_run, resumed_run = workflow_spans
    for record in _events(log_exporter, PAUSED) + [
        record
        for record in _events(log_exporter, CHECKPOINTED)
        if record.attributes["gen_ai.agent.checkpoint.id"]
        in first_run_checkpoints
    ]:
        assert record.trace_id == paused_run.context.trace_id
        assert record.span_id == paused_run.context.span_id
    for record in _events(log_exporter, RESUMED) + [
        record
        for record in _events(log_exporter, CHECKPOINTED)
        if record.attributes["gen_ai.agent.checkpoint.id"]
        in second_run_checkpoints
    ]:
        assert record.trace_id == resumed_run.context.trace_id
        assert record.span_id == resumed_run.context.span_id

    # The two invoke calls are separate traces, and nothing in the telemetry
    # ties them together: LangGraph mints no end-to-end execution id.
    assert paused_run.context.trace_id != resumed_run.context.trace_id


def test_graph_without_checkpointer_emits_no_durability_events(
    start_instrumentation,
    log_exporter,
):
    graph = _build_graph().compile()

    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "x"}})

    assert _events(log_exporter, CHECKPOINTED) == []
    assert _events(log_exporter, RESUMED) == []
    # The interrupt still happens, so paused is still reported. LangGraph
    # supplies a checkpoint id for the in-memory loop checkpoint even though
    # nothing persisted it.
    assert len(_events(log_exporter, PAUSED)) == 1


def test_plain_graph_run_emits_no_lifecycle_events(
    start_instrumentation,
    log_exporter,
):
    builder = StateGraph(_State)
    builder.add_node("only", lambda _state: {"value": "done"})
    builder.add_edge(START, "only")
    builder.add_edge("only", END)
    graph = builder.compile()

    assert graph.invoke({"value": "new"}) == {"value": "done"}

    for event_name in LIFECYCLE_EVENTS:
        assert _events(log_exporter, event_name) == []


def test_async_interrupt_and_resume_emit_lifecycle_events(
    start_instrumentation,
    log_exporter,
):
    graph = _build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "async-lifecycle-test"}}

    async def run() -> str:
        paused_result = await graph.ainvoke({"value": "new"}, config=config)
        await graph.ainvoke(Command(resume="approved"), config=config)
        return paused_result["__interrupt__"][0].id

    real_interrupt_id = asyncio.run(run())

    paused = _events(log_exporter, PAUSED)
    assert len(paused) == 1
    assert paused[0].attributes["gen_ai.agent.pause.id"] == real_interrupt_id

    # aput is wrapped alongside put, so the per-superstep volume matches the
    # synchronous run.
    assert len(_events(log_exporter, CHECKPOINTED)) == 5
    resumed = _events(log_exporter, RESUMED)
    assert len(resumed) == 1
    assert (
        resumed[0].attributes["gen_ai.agent.resumed_from.id"]
        == paused[0].attributes["gen_ai.agent.checkpoint.id"]
    )


def test_uninstrument_restores_the_checkpointer(
    start_instrumentation,
    log_exporter,
):
    checkpointer = InMemorySaver()
    graph = _build_graph().compile(checkpointer=checkpointer)
    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "u"}})
    assert _events(log_exporter, CHECKPOINTED)

    start_instrumentation.uninstrument()

    assert "put" not in checkpointer.__dict__
    assert "aput" not in checkpointer.__dict__


# ---------------------------------------------------------------------------
# Correlation and checkpointer wrapping
# ---------------------------------------------------------------------------


class _RecordingSaver(InMemorySaver):
    """Saver that records every checkpoint it actually persists."""

    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[tuple[str, str]] = []

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        configurable = result["configurable"]
        self.recorded.append(
            (
                str(config.get("configurable", {}).get("checkpoint_ns", "")),
                str(configurable["checkpoint_id"]),
            )
        )
        return result


def _checkpoint_ids(log_exporter) -> list[str]:
    return [
        record.attributes["gen_ai.agent.checkpoint.id"]
        for record in _events(log_exporter, CHECKPOINTED)
    ]


def test_nested_subgraph_checkpoints_correlate_to_the_owning_workflow(
    start_instrumentation,
    log_exporter,
    span_exporter,
):
    class _SubState(TypedDict, total=False):
        decision: str

    sub_builder = StateGraph(_SubState)
    sub_builder.add_node(
        "ask", lambda _state: {"decision": str(interrupt("approve?"))}
    )
    sub_builder.add_edge(START, "ask")
    sub_builder.add_edge("ask", END)

    builder = StateGraph(_State)
    builder.add_node("prepare", lambda _state: {"value": "prepared"})
    builder.add_node("child", sub_builder.compile())
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "child")
    builder.add_edge("child", END)

    saver = _RecordingSaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "nested-test"}}

    graph.invoke({"value": "new"}, config=config)
    first_run_writes = list(saver.recorded)
    graph.invoke(Command(resume="approved"), config=config)

    # Both the root namespace and the subgraph namespace are exercised.
    namespaces = {namespace for namespace, _ in saver.recorded}
    assert "" in namespaces
    assert any(namespace.startswith("child:") for namespace in namespaces)

    # Every checkpoint LangGraph persisted is reported exactly once, at every
    # nesting level, and none is invented.
    assert _checkpoint_ids(log_exporter) == [
        checkpoint_id for _, checkpoint_id in saver.recorded
    ]

    # LangGraph classifies a subgraph's chain run as a plain chain, not a
    # workflow, so no child workflow span exists and the child namespace's
    # checkpoints resolve to the nearest enclosing run: the parent workflow.
    # Namespace resolution itself is covered in test_invocation_manager.
    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2
    first_ids = {checkpoint_id for _, checkpoint_id in first_run_writes}
    for record in _events(log_exporter, CHECKPOINTED):
        expected = (
            workflow_spans[0]
            if record.attributes["gen_ai.agent.checkpoint.id"] in first_ids
            else workflow_spans[1]
        )
        assert record.span_id == expected.context.span_id
        assert record.trace_id == expected.context.trace_id


def test_two_unrelated_runs_on_one_thread_id_drop_the_checkpoint(
    start_instrumentation,
    log_exporter,
):
    from unittest import mock
    from uuid import uuid4

    from opentelemetry.instrumentation.genai.langchain.callback_handler import (
        OpenTelemetryLangChainCallbackHandler,
    )
    from opentelemetry.util.genai.invocation import WorkflowInvocation

    telemetry = mock.MagicMock()
    telemetry.should_capture_content.return_value = False
    telemetry.workflow.side_effect = lambda **_kwargs: mock.MagicMock(
        spec=WorkflowInvocation
    )
    handler = OpenTelemetryLangChainCallbackHandler(telemetry)

    metadata = {"ls_integration": "langgraph", "thread_id": "shared"}
    workflows = []
    for _ in range(2):
        run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            metadata=metadata,
        )
        workflows.append(handler._invocation_manager.get_invocation(run_id))

    handler.checkpoint_written("shared", "ckpt-1")

    for workflow in workflows:
        workflow.emit_event.assert_not_called()


class _DelegatingSaver(InMemorySaver):
    """Saver whose ``aput`` delegates to ``put``, like ``InMemorySaver``."""

    def __init__(self) -> None:
        super().__init__()
        self.write_count = 0

    def put(self, config, checkpoint, metadata, new_versions):
        self.write_count += 1
        return super().put(config, checkpoint, metadata, new_versions)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return self.put(config, checkpoint, metadata, new_versions)


class _WorkerThreadSaver(_DelegatingSaver):
    """Saver whose ``aput`` runs ``put`` on a worker thread.

    The worker gets no copy of the caller's context, so de-duplication cannot
    rely on context propagation.
    """

    async def aput(self, config, checkpoint, metadata, new_versions):
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return await loop.run_in_executor(
                pool,
                lambda: self.put(config, checkpoint, metadata, new_versions),
            )


@pytest.mark.parametrize(
    "saver_factory", [_DelegatingSaver, _WorkerThreadSaver]
)
def test_delegating_saver_reports_each_write_once(
    start_instrumentation,
    log_exporter,
    saver_factory,
):
    saver = saver_factory()
    graph = _build_graph().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "delegating-test"}}

    asyncio.run(graph.ainvoke({"value": "new"}, config=config))

    checkpoint_ids = _checkpoint_ids(log_exporter)
    assert saver.write_count == 3
    assert len(checkpoint_ids) == saver.write_count
    assert len(set(checkpoint_ids)) == saver.write_count


def test_uninstrument_restores_instance_level_put_and_aput(
    start_instrumentation,
    log_exporter,
):
    saver = InMemorySaver()
    original_put = saver.put
    original_aput = saver.aput
    # A saver that legitimately supplies its write methods per instance.
    saver.put = original_put
    saver.aput = original_aput

    graph = _build_graph().compile(checkpointer=saver)
    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "i"}})
    assert _events(log_exporter, CHECKPOINTED)

    start_instrumentation.uninstrument()

    assert saver.__dict__["put"] is original_put
    assert saver.__dict__["aput"] is original_aput


def test_untrackable_saver_is_left_untouched(
    start_instrumentation,
    log_exporter,
):
    class _UnhashableSaver(InMemorySaver):
        # Defining __eq__ without __hash__ makes instances unhashable, so the
        # saver cannot be registered for restoration.
        def __eq__(self, other: object) -> bool:
            return self is other

    saver = _UnhashableSaver()
    graph = _build_graph().compile(checkpointer=saver)

    # Nothing was patched, so nothing is left behind to clean up.
    assert "put" not in saver.__dict__
    assert "aput" not in saver.__dict__

    graph.invoke({"value": "new"}, config={"configurable": {"thread_id": "n"}})

    assert _events(log_exporter, CHECKPOINTED) == []


def test_checkpointer_is_discovered_from_the_compiled_graph(
    start_instrumentation,
    log_exporter,
):
    keyword_saver = InMemorySaver()
    keyword_graph = _build_graph().compile(checkpointer=keyword_saver)
    positional_saver = InMemorySaver()
    positional_graph = _build_graph().compile(positional_saver)

    # The saver the compiled graph retains is the one that was patched.
    assert keyword_graph.checkpointer is keyword_saver
    assert positional_graph.checkpointer is positional_saver
    assert "put" in keyword_saver.__dict__
    assert "put" in positional_saver.__dict__

    keyword_graph.invoke(
        {"value": "new"}, config={"configurable": {"thread_id": "kw"}}
    )
    keyword_checkpoints = len(_checkpoint_ids(log_exporter))
    positional_graph.invoke(
        {"value": "new"}, config={"configurable": {"thread_id": "pos"}}
    )

    assert keyword_checkpoints == 3
    assert len(_checkpoint_ids(log_exporter)) == 6


def test_nested_resume_reports_one_event_per_graph_level(
    start_instrumentation,
    log_exporter,
):
    class _SubState(TypedDict, total=False):
        decision: str

    sub_builder = StateGraph(_SubState)
    sub_builder.add_node(
        "ask", lambda _state: {"decision": str(interrupt("approve?"))}
    )
    sub_builder.add_edge(START, "ask")
    sub_builder.add_edge("ask", END)

    builder = StateGraph(_State)
    builder.add_node("prepare", lambda _state: {"value": "prepared"})
    builder.add_node("child", sub_builder.compile())
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "child")
    builder.add_edge("child", END)

    saver = _RecordingSaver()
    graph = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "nested-resume"}}

    graph.invoke({"value": "new"}, config=config)
    persisted = {
        namespace: checkpoint_id for namespace, checkpoint_id in saver.recorded
    }
    graph.invoke(Command(resume="approved"), config=config)

    resumed_ids = [
        record.attributes["gen_ai.agent.resumed_from.id"]
        for record in _events(log_exporter, RESUMED)
    ]
    # One resumed event per graph level: the root graph and the subgraph.
    assert len(resumed_ids) == 2
    assert len(set(resumed_ids)) == 2

    # Provenance: each id is the last checkpoint that level actually persisted.
    root_namespace = ""
    child_namespace = next(
        namespace for namespace in persisted if namespace.startswith("child:")
    )
    assert set(resumed_ids) == {
        persisted[root_namespace],
        persisted[child_namespace],
    }
    assert all(
        record.attributes["gen_ai.agent.resumed_from.type"] == "checkpoint"
        for record in _events(log_exporter, RESUMED)
    )


# ---------------------------------------------------------------------------
# Checkpoint write de-duplication
# ---------------------------------------------------------------------------


class _StubReporter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()

    def checkpoint_written(
        self, thread_id: str, checkpoint_id: str, namespace: str
    ) -> None:
        with self._lock:
            self.calls.append((thread_id, checkpoint_id, namespace))


class _FixedIdSaver:
    """Minimal saver whose writes always persist the same checkpoint id."""

    def __init__(self, checkpoint_id: str = "fixed-ckpt") -> None:
        self._checkpoint_id = checkpoint_id
        self.write_count = 0

    def put(self, config, checkpoint, metadata, new_versions):
        self.write_count += 1
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get(
                    "checkpoint_ns", ""
                ),
                "checkpoint_id": self._checkpoint_id,
            }
        }

    async def aput(self, config, checkpoint, metadata, new_versions):
        # Delegation, passing the same checkpoint object through.
        return self.put(config, checkpoint, metadata, new_versions)


def _fixed_id_config(thread_id: str = "dedup"):
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def test_delegated_write_is_reported_once_by_the_outer_call():
    saver = _FixedIdSaver()
    reporter = _StubReporter()
    _wrap_checkpointer(saver, reporter)

    asyncio.run(saver.aput(_fixed_id_config(), {"id": "a"}, {}, {}))

    assert saver.write_count == 1
    assert reporter.calls == [("dedup", "fixed-ckpt", "")]


def test_separate_writes_reusing_a_checkpoint_id_are_both_reported():
    saver = _FixedIdSaver()
    reporter = _StubReporter()
    _wrap_checkpointer(saver, reporter)

    saver.put(_fixed_id_config(), {"id": "a"}, {}, {})
    saver.put(_fixed_id_config(), {"id": "b"}, {}, {})

    # Two distinct writes, so two events even though the id is identical.
    assert saver.write_count == 2
    assert len(reporter.calls) == 2


class _SequencedSaver:
    """Saver that lets a test force an exact interleaving of two writes.

    ``aput`` delegates to ``put`` with the same checkpoint object, then holds
    the outer call open until a second, independent write has entered.
    """

    def __init__(self) -> None:
        self.write_count = 0
        self.delegated_returned = threading.Event()
        self.second_write_entered = threading.Event()
        self._lock = threading.Lock()

    def put(self, config, checkpoint, metadata, new_versions):
        with self._lock:
            self.write_count += 1
        if checkpoint["name"] == "B":
            self.second_write_entered.set()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": "",
                "checkpoint_id": f"ckpt-{checkpoint['name']}",
            }
        }

    async def aput(self, config, checkpoint, metadata, new_versions):
        result = self.put(config, checkpoint, metadata, new_versions)
        self.delegated_returned.set()
        # Hold the outer write open until the independent write has entered.
        assert self.second_write_entered.wait(10)
        return result


def test_forced_interleaving_reports_each_write_exactly_once():
    saver = _SequencedSaver()
    reporter = _StubReporter()
    _wrap_checkpointer(saver, reporter)
    config = _fixed_id_config("shared")

    def write_a() -> None:
        asyncio.run(saver.aput(config, {"name": "A"}, {}, {}))

    def write_b() -> None:
        # Enter only after A's delegated inner call has already returned.
        assert saver.delegated_returned.wait(10)
        saver.put(config, {"name": "B"}, {}, {})

    threads = [
        threading.Thread(target=write_a),
        threading.Thread(target=write_b),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not any(thread.is_alive() for thread in threads)

    # Sequence forced above: outer A enters, delegated A enters and returns,
    # outer B enters, outer A returns, outer B returns. The delegation is
    # silent and neither independent write is dropped as a false duplicate.
    assert saver.write_count == 2
    assert sorted(checkpoint_id for _, checkpoint_id, _ in reporter.calls) == [
        "ckpt-A",
        "ckpt-B",
    ]


def test_concurrent_writes_through_a_real_delegating_saver(
    start_instrumentation,
    log_exporter,
):
    saver = _DelegatingSaver()
    graph = _build_graph().compile(checkpointer=saver)
    started = threading.Barrier(4)

    def run(index: int) -> None:
        started.wait()
        graph.invoke(
            {"value": "new"},
            config={"configurable": {"thread_id": f"concurrent-{index}"}},
        )

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    checkpoint_ids = _checkpoint_ids(log_exporter)
    assert saver.write_count == 12
    assert len(checkpoint_ids) == saver.write_count
    assert len(set(checkpoint_ids)) == saver.write_count

OpenTelemetry LangGraph agent lifecycle example
===============================================

This is an example of the agent lifecycle telemetry the instrumentation
captures from a LangGraph durable execution: a graph that pauses on
``interrupt()``, checkpoints, and resumes on a second invoke.

`main.py <main.py>`_ needs no API key and no collector. It runs the graph twice
under the instrumentor with in-memory exporters and writes every span and log
record it produced to `sample-output.json <sample-output.json>`_, which is the
captured run committed here.

Setup
-----

Set up a virtual environment like this:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run
---

Run the example from the repository root like this:

::

    python instrumentation/opentelemetry-instrumentation-genai-langchain/examples/langgraph-lifecycle/main.py

You should see ``gen_ai.agent.paused``, ``gen_ai.agent.checkpointed``, and
``gen_ai.agent.resumed`` events correlated with the workflow span of the invoke
that produced them.

Design note
-----------

Every name below is a candidate semantic convention proposed in
open-telemetry/semantic-conventions-genai#445, and none of them is stable.
Everything stated here was measured against langgraph 1.2.9 and
langchain-core 1.5.0.

Interception points
~~~~~~~~~~~~~~~~~~~

1. ``langgraph.callbacks`` lifecycle dispatch. LangGraph 1.2 calls
   ``on_interrupt`` and ``on_resume`` on the handlers in a run's callback
   manager, passing ``GraphInterruptEvent`` and ``GraphResumeEvent`` with the
   real ``Interrupt`` objects, the checkpoint id, and the top level run id. The
   released instrumentor is already in that handler list: without these two
   methods LangGraph logs ``AttributeError`` on every interrupt and resume, so
   adding them also removes existing log noise.
2. ``StateGraph.compile(checkpointer=...)``. The saver the compiled graph
   retains is wrapped so every persisted checkpoint reports its id.
   ``BaseCheckpointSaver.put`` is abstract and every saver overrides it, so the
   instance is patched rather than the base class. ``aput`` may delegate to
   ``put``, possibly on a worker thread, so a write is de-duplicated by holding
   the identity of the ``Checkpoint`` object while the write is in flight, not
   by remembering ids and not by relying on context propagation: a nested call
   receiving the same object is the delegation and stays silent, and the
   outermost call reports. This assumes one checkpoint object per logical
   write, which LangGraph guarantees by handing the saver a freshly constructed
   mapping per superstep (``_loop.py`` passes
   ``copy_checkpoint(self.checkpoint)``), so two independent writes never share
   an object. A saver that copies the checkpoint before delegating is reported
   twice, which is the safe direction.

The interrupt is not observable from ``on_chain_end``: LangGraph adds
``__interrupt__`` to the invoke return value after the callback fires.

Correlation
~~~~~~~~~~~

``paused`` and ``resumed`` use the run id on the lifecycle event, resolved to
the nearest workflow invocation. A checkpointer call has no run id, so
``checkpointed`` maps the ``configurable.thread_id`` and
``configurable.checkpoint_ns`` in its config to a live run. Runs are tracked per
thread id, each bound with the namespace its own checkpoints use ("" for a top
level graph, otherwise the run's ``langgraph_checkpoint_ns``). A write resolves
to the run bound with that exact namespace, or to the nearest enclosing run when
none is. A nested run never displaces the run containing it, and when two
equally specific live runs are unrelated the event is dropped rather than
guessed.

LangGraph classifies a subgraph's own graph run as a plain chain, not a
workflow, so today the child namespace resolves to the parent workflow. The
resolution is namespace aware regardless, so a nested workflow would own its own
writes.

``resumed_from.type`` is always ``checkpoint``. That is a constant in the
instrumentation, but it is determined by LangGraph, not chosen: the resume event
supplies a checkpoint id and nothing else.

``GenAIInvocation.emit_event`` is byte-identical to the hunk in open PR #507,
saved as ``util/opentelemetry-util-genai/tests/fixtures/pr507_emit_event.py.txt``
and checked by a test, and is dropped when rebasing onto that PR. It is not a
competing API.

Checkpoint volume
~~~~~~~~~~~~~~~~~

``put`` fires once per superstep. The captured run emits 3 ``checkpointed``
events for the invoke that pauses, 2 for the invoke that resumes. Nested
subgraphs checkpoint independently, under their own namespace, and emit one
``resumed`` per graph level, which the flat event model does not distinguish.

Not demonstrable
~~~~~~~~~~~~~~~~

``gen_ai.agent.execution.id`` is omitted. LangGraph mints no id spanning suspend
and resume: ``thread_id`` is a conversation reused across runs, and every other
id changes on the resuming invoke. The captured run shows the two invokes as
separate traces with nothing linking them, which is the gap the attribute
describes.

``gen_ai.agent.pause.reason`` is omitted. ``interrupt(value)`` carries an opaque
application payload and nothing saying who resolves the pause, so neither
``human_input`` nor ``external_system`` is derivable. The ``pause`` member of
``resumed_from.type`` is likewise absent: LangGraph never reports a pause id at
resume time.

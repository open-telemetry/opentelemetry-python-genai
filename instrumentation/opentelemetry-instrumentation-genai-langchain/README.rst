OpenTelemetry LangChain Instrumentation
=======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-langchain.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-langchain/

This library traces `LangChain <https://pypi.org/project/langchain/>`_ and
`LangGraph <https://pypi.org/project/langgraph/>`_ applications. It hooks into
LangChain's callback manager to emit spans that mirror the structure of your
application:

* **Workflow spans** for a graph or chain run — for example an invocation of a
  LangGraph ``StateGraph`` — capturing the overall input and output of the run.
* **Agent spans** for agent invocations nested inside a workflow, including the
  agent name, id, description, and conversation/session id when available.
* **Tool spans** for tool calls made during a run.

The spans nest to reflect the graph, so a single graph invocation produces a
workflow span with the agent, tool, and model calls it triggered as children.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-langchain

See the `examples <examples>`_ directory for runnable ``workflow``, ``agent``,
``tools``, and ``zero-code`` scenarios.

Usage
-----

Call ``LangChainInstrumentor().instrument()`` once during startup, then build
and invoke your graph as usual. The example below traces a simple two-node
LangGraph ``StateGraph`` (``START → researcher → summariser → END``); the
``graph.invoke(...)`` call is recorded as a workflow span with the node model
calls nested underneath.

.. code-block:: python

    from typing import Annotated, TypedDict

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="<your-model>", temperature=0)


    class State(TypedDict):
        messages: Annotated[list, add_messages]
        research: str


    def researcher(state: State) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content="Provide 2-3 factual sentences."),
                HumanMessage(content=state["messages"][-1].content),
            ]
        )
        return {"research": response.content, "messages": [response]}


    def summariser(state: State) -> dict:
        response = llm.invoke(
            [
                SystemMessage(content="Condense the text into one sentence."),
                HumanMessage(content=state["research"]),
            ]
        )
        return {"messages": [response]}


    builder = StateGraph(State)
    builder.add_node("researcher", researcher)
    builder.add_node("summariser", summariser)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "summariser")
    builder.add_edge("summariser", END)
    graph = builder.compile()

    # Recorded as a workflow span with the two node LLM calls nested underneath.
    graph.invoke(
        {
            "messages": [HumanMessage(content="What is the capital of France?")],
            "research": "",
        }
    )

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `LangGraph <https://langchain-ai.github.io/langgraph/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.genai.langchain import (
    LangChainInstrumentor,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.test_util_genai.instrumentor import instrument

_PROMPT_VERSION = "gen_ai.prompt.version"


@pytest.fixture
def instrumented(tracer_provider, meter_provider, logger_provider):
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        yield


def _model() -> FakeMessagesListChatModel:
    return FakeMessagesListChatModel(responses=[AIMessage(content="ok")])


def _call(model_name: str) -> dict:
    return {"metadata": {"ls_model_name": model_name}}


def _prompt_attributes(span_exporter) -> dict[str, dict]:
    return {
        span.attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL]: {
            key: value
            for key, value in span.attributes.items()
            if key.startswith("gen_ai.prompt.")
        }
        for span in span_exporter.get_finished_spans()
        if span.attributes.get(gen_ai_attributes.GEN_AI_OPERATION_NAME)
        == "chat"
    }


def _greeting() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([("human", "Say hi to {who}.")])
    prompt.name = "greeting"
    prompt.metadata = {"prompt_version": "3"}
    return prompt


_GREETING = {
    gen_ai_attributes.GEN_AI_PROMPT_NAME: "greeting",
    _PROMPT_VERSION: "3",
    "gen_ai.prompt.variable.who": "Ada",
}


@pytest.mark.usefixtures("instrumented")
def test_prompt_formatted_outside_a_chain(span_exporter):
    prompt_value = _greeting().invoke({"who": "Ada"})
    _model().invoke(prompt_value, config=_call("invoke"))
    list(_model().stream(prompt_value, config=_call("stream")))

    assert _prompt_attributes(span_exporter) == {
        "invoke": _GREETING,
        "stream": _GREETING,
    }


@pytest.mark.usefixtures("instrumented")
def test_async_prompt_formatted_outside_a_chain(span_exporter):
    async def run() -> None:
        prompt_value = await _greeting().ainvoke({"who": "Ada"})
        await _model().ainvoke(prompt_value, config=_call("ainvoke"))
        async for _ in _model().astream(prompt_value, config=_call("astream")):
            pass

    asyncio.run(run())

    assert _prompt_attributes(span_exporter) == {
        "ainvoke": _GREETING,
        "astream": _GREETING,
    }


@pytest.mark.usefixtures("instrumented")
def test_each_model_call_gets_the_prompt_that_built_its_input(span_exporter):
    summary = PromptTemplate.from_template("Summarize: {text}", name="summary")

    def node(_: dict) -> str:
        # Both prompts run under this node before either model call.
        greeting_value = _greeting().invoke({"who": "Ada"})
        summary_value = summary.invoke({"text": "OTel"})
        _model().invoke(summary_value, config=_call("summary-call"))
        _model().invoke(greeting_value, config=_call("greeting-call"))
        return "done"

    RunnableLambda(node).invoke({})

    assert _prompt_attributes(span_exporter) == {
        "summary-call": {
            gen_ai_attributes.GEN_AI_PROMPT_NAME: "summary",
            "gen_ai.prompt.variable.text": "OTel",
        },
        "greeting-call": _GREETING,
    }


@pytest.mark.usefixtures("instrumented")
def test_hub_prompt_reports_repo_and_commit(span_exporter):
    prompt = ChatPromptTemplate.from_messages([("human", "Weather in {city}?")])
    prompt.metadata = {
        "lc_hub_owner": "acme",
        "lc_hub_repo": "weather-now",
        "lc_hub_commit_hash": "9f2c1ab",
    }

    (prompt | _model()).invoke({"city": "Oslo"}, config=_call("hub"))

    assert _prompt_attributes(span_exporter)["hub"] == {
        gen_ai_attributes.GEN_AI_PROMPT_NAME: "weather-now",
        _PROMPT_VERSION: "9f2c1ab",
        "gen_ai.prompt.variable.city": "Oslo",
    }


class _ReviewPrompt(ChatPromptTemplate):
    pass


@pytest.mark.usefixtures("instrumented")
def test_prompt_template_subclass_is_detected(span_exporter):
    named = _ReviewPrompt.from_messages([("human", "Review {code}")])
    named.name = "code-review"
    unnamed = _ReviewPrompt.from_messages([("human", "Review {code}")])
    unnamed.metadata = {"prompt_version": "2"}

    (named | _model()).invoke({"code": "x = 1"}, config=_call("named"))
    (unnamed | _model()).invoke({"code": "x = 1"}, config=_call("unnamed"))

    attributes = _prompt_attributes(span_exporter)
    assert attributes["named"] == {
        gen_ai_attributes.GEN_AI_PROMPT_NAME: "code-review",
        "gen_ai.prompt.variable.code": "x = 1",
    }
    # A version names nothing without a prompt name.
    assert attributes["unnamed"] == {"gen_ai.prompt.variable.code": "x = 1"}


@pytest.mark.usefixtures("instrumented")
def test_model_input_not_built_from_a_prompt(span_exporter):
    _greeting().invoke({"who": "Ada"})
    _model().invoke("Plain message.", config=_call("plain"))

    assert _prompt_attributes(span_exporter) == {"plain": {}}

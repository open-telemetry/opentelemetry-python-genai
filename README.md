# <img src="https://opentelemetry.io/img/logos/opentelemetry-logo-nav.png" alt="OpenTelemetry Icon" width="45"> OpenTelemetry Python GenAI Instrumentations

OpenTelemetry instrumentation packages for Generative AI client libraries and frameworks in
Python.

This repo is built on top of the OpenTelemetry [Python SDK](https://opentelemetry.io/docs/languages/python/) and the [opentelemetry-instrumentation](https://github.com/open-telemetry/opentelemetry-python-contrib/tree/main/opentelemetry-instrumentation) package hosted in the [opentelemetry-python-contrib](https://github.com/open-telemetry/opentelemetry-python-contrib) repo.

All instrumentations use [opentelemetry-util-genai](./util/opentelemetry-util-genai) and emit spans, metrics, and logs according to [GenAI Semantic Conventions](https://github.com/open-telemetry/semantic-conventions-genai).


<!-- instrumentations -->
## Released Instrumentations

| Instrumentation | Supported Package | Version |
| --------------- | ----------------- | ------- |
| [opentelemetry-instrumentation-genai-agno](./instrumentation/opentelemetry-instrumentation-genai-agno) | agno >= 2.0.0, < 3 | [1.1b0](https://pypi.org/project/opentelemetry-instrumentation-genai-agno/) |
| [opentelemetry-instrumentation-genai-anthropic](./instrumentation/opentelemetry-instrumentation-genai-anthropic) | anthropic >= 0.51.0, < 2 | [1.1b1](https://pypi.org/project/opentelemetry-instrumentation-genai-anthropic/) |
| [opentelemetry-instrumentation-genai-langchain](./instrumentation/opentelemetry-instrumentation-genai-langchain) | langchain >= 0.3.21, < 2 | [1.1b1](https://pypi.org/project/opentelemetry-instrumentation-genai-langchain/) |
| [opentelemetry-instrumentation-genai-openai](./instrumentation/opentelemetry-instrumentation-genai-openai) | openai >= 1.26.0, < 4 | [1.1b0](https://pypi.org/project/opentelemetry-instrumentation-genai-openai/) |
| [opentelemetry-instrumentation-genai-openai-agents](./instrumentation/opentelemetry-instrumentation-genai-openai-agents) | openai-agents >= 0.3.3, < 1 | [1.1b0](https://pypi.org/project/opentelemetry-instrumentation-genai-openai-agents/) |
| [opentelemetry-instrumentation-genai-qwen-agent](./instrumentation/opentelemetry-instrumentation-genai-qwen-agent) | qwen-agent >= 0.0.20, < 1 | [1.1b0](https://pypi.org/project/opentelemetry-instrumentation-genai-qwen-agent/) |
| [opentelemetry-instrumentation-genai-smolagents](./instrumentation/opentelemetry-instrumentation-genai-smolagents) | smolagents >= 1.24.0, < 2 | [1.1b0](https://pypi.org/project/opentelemetry-instrumentation-genai-smolagents/) |
| [opentelemetry-instrumentation-google-genai](./instrumentation/opentelemetry-instrumentation-google-genai) | google-genai >= 1.32.0, <3 | [1.1b1](https://pypi.org/project/opentelemetry-instrumentation-google-genai/) |

## Unreleased Instrumentations

| Instrumentation | Supported Package | Version | Status |
| --------------- | ----------------- | ------- | ------ |
| [opentelemetry-instrumentation-genai-claude-agent-sdk](./instrumentation/opentelemetry-instrumentation-genai-claude-agent-sdk) | claude-agent-sdk >= 0.1.14, < 1 | 1.2b0.dev | skeleton |
| [opentelemetry-instrumentation-genai-crewai](./instrumentation/opentelemetry-instrumentation-genai-crewai) | crewai >= 1.10.1, < 2 | 1.2b0.dev | skeleton |
| [opentelemetry-instrumentation-genai-llama-index](./instrumentation/opentelemetry-instrumentation-genai-llama-index) | llama-index-core >= 0.14.19, < 1 | 1.2b0.dev | skeleton |
| [opentelemetry-instrumentation-genai-weaviate-client](./instrumentation/opentelemetry-instrumentation-genai-weaviate-client) | weaviate-client >= 3.0.0, <5.0.0 | 1.2b0.dev | skeleton |
<!-- end -->

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

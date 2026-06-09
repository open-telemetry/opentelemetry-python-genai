OpenTelemetry-Python-GenAI
==========================

OpenTelemetry instrumentation and shared utilities for Generative AI client
libraries in Python.

.. image:: https://img.shields.io/badge/slack-chat-green.svg
   :target: https://cloud-native.slack.com/archives/C01PD4HUVBL
   :alt: Slack Chat


**Please note** that this library is currently in _beta_, and shouldn't
generally be used in production environments.

Installation
------------

GenAI instrumentation packages are available on PyPI and can be installed
separately via pip:

.. code-block:: sh

    pip install opentelemetry-instrumentation-genai-{instrumentation}

A complete list of packages can be found in the
`opentelemetry-python-genai instrumentation <https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/instrumentation>`_
directory.

Installing Cutting Edge Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the project is pre-1.0, there may be significant functionality that
has not yet been released to PyPI. In that situation, you may want to
install the packages directly from the repo. This can be done by cloning the
repository and doing an `editable
install <https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs>`_:

.. code-block:: sh

    git clone https://github.com/open-telemetry/opentelemetry-python-genai.git
    cd opentelemetry-python-genai
    pip install -e ./util/opentelemetry-util-genai
    pip install -e ./instrumentation/opentelemetry-instrumentation-genai-openai
    pip install -e ./instrumentation/opentelemetry-instrumentation-genai-anthropic


.. toctree::
    :maxdepth: 2
    :caption: OpenTelemetry GenAI Instrumentations
    :name: GenAI Instrumentations
    :glob:

    instrumentation/**

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

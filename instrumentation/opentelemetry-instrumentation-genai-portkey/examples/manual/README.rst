Portkey AI Manual Instrumentation Example
=========================================

This example demonstrates how to instrument the `Portkey AI <https://github.com/Portkey-AI/portkey-python-sdk>`_ SDK with OpenTelemetry manually.

Installation
------------

Install the required dependencies:

::

    pip install -r requirements.txt

Running the Example
-------------------

Set your Portkey API key:

::

    export PORTKEY_API_KEY="your-portkey-api-key"

Run the basic example:

::

    python main.py

Run the custom completion hook example:

::

    python custom_hook.py

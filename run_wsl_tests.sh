#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
export UV_LINK_MODE=copy
uv run tox

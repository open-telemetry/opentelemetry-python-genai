# Review rules for all PRs

## 0. What not to flag

CI already runs deterministic gates on every PR. If a gate would catch it, the gate reports it, 
and a duplicate review comment is noise:

| Gate | Covers |
| --- | --- |
| `precommit` (ruff, ruff-format, uv-lock, rstcheck) | Lint rules, formatting, import order, line length, `uv.lock` freshness, RST syntax |
| `typecheck` (pyright, strict) | Type errors and unused `type: ignore` in packages listed under `[tool.pyright] include` |
| `spellcheck` (codespell) | Spelling in code, docs, and comments |
| `shellcheck` | Shell script correctness in `scripts/*.sh` |
| `readme` | `README.rst` renders on PyPI |
| `docs` | Docs build |
| `generate` | The generated instrumentation README table being up to date |
| `lint-license-header-check` | Missing license headers |
| `deps-check` | Package.py _instruments matching pyproject.toml, redundant pins in `tests/requirements.oldest.txt`, packages missing an `oldest` requirements file |
| `changelog` workflow | Direct edits to `CHANGELOG.md`, changelog fragment presence |
| `test` matrix | Test failures across the `oldest`/`latest`/`conformance` envs |

Review the things a gate cannot decide: whether the change is correct, whether it matches the
GenAI semantic conventions, whether it actually achieves what the PR claims, and whether it
weakens or bypasses a gate. Style opinions that ruff does not enforce are not review comments.

## 1. Shared configuration

Repo-wide config is `tox.ini`, `pytest.ini`, root `pyproject.toml`, `.codespellrc`,
`.pre-commit-config.yaml`, and `scripts/`. A change there affects every package, so:

- **No developer-local paths.** Personal venv or scratch directory names must not be added to
  shared configs. `.gitignore` already covers `venv*/`, `.venv*/`, and `.tox`; anything else
  belongs in a local ignore or gets renamed to match those patterns. Flag entries that only make
  sense on one contributor's machine.
- **Skip lists must not weaken a gate.** Excluding a bare directory name from a lint or license
  check also excludes any real source directory that happens to share that name. Prefer
  path-anchored patterns (`*/.tox/*`) over bare components (`.tox`).
- **Shell must be portable.** Commands in `tox.ini` and `scripts/*.sh` run on both Linux and
  macOS. Flag GNU-only flags (e.g. `xargs --no-run-if-empty`, `sed -i` without a backup suffix,
  `readlink -f`).

## 2. Config that packages can shadow

Root config does not automatically reach every package. Before approving a change to a root
setting, check whether packages override the same section in their own `pyproject.toml`:

- `pytest.ini` is only read when pytest's config lookup does not stop at a package
  `pyproject.toml` first. These packages define `[tool.pytest.ini_options]` and therefore ignore
  root `pytest.ini` entirely: `anthropic`, `agno`, `crewai`, `smolagents`, `llama-index`,
  `claude-agent-sdk`. A new root pytest setting must either be added to each of them, or those
  blocks removed.
- The same applies to `[tool.ruff]` and `[tool.pyright]` overrides.

Flag a root-config change whose stated goal ("fixes the warning everywhere", "applies to all
tests") is not actually achieved for the shadowing packages.

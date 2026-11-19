# Review rules for all PRs

Report problems introduced or made worse by the diff. Before reporting, identify the changed
line, the contract or repository rule it violates, and the concrete consequence. Check nearby
code and existing review comments; don't repeat findings or flag unrelated pre-existing issues.
For missing tests, name the behavior that could regress without an assertion.

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
weakens or bypasses a gate. Formatting is not review comments. Dead code,
duplication, stale or restating comments, and untyped code are.

## Top checks (all PRs)

- **Duplication.** Flag blocks of ~5+ lines copy-pasted across modules, wrappers, or sync/async
  variants, and helpers re-implemented when the package or `opentelemetry-util-genai` already
  has one.
- **Dead code.** Flag unused parameters, fields, aliases, imports, and helpers; tests that only
  exercise dead code; and attributes set again after being passed to a `handler.*()` factory.
  ```python
  # Flag
  inv = handler.inference(provider, model, server_address=host)
  inv.request_model = model  # already passed to the factory
  ```
- **Comments and docs.** Flag comments that restate the code, describe history ("previously",
  "now we"), justify the change, or no longer match the code.
  ```python
  # Flag
  # Set the model on the invocation
  invocation.request_model = model
  ```
- **Types.** Flag `Any`/`object` annotations and `getattr`/`hasattr` on objects whose type is
  known. Pyright does not catch these.
- **User-facing docs** (`README.rst`, public API docstrings, changelog fragments). Flag text
  that is verbose, mentions private names or implementation details, contains boilerplate, or no
  longer matches the behavior. Changelog fragments are one short line; flag fragments for
  test-only/tooling/internal changes (use the `Skip Changelog` label instead).
- **Scope.** Flag changes unrelated to the PR's stated goal, reverts of recently merged unrelated
  changes (stale branch), and titles/descriptions/changelogs that don't match the diff.
- **Version-specific code.** Flag branches for specific dependency versions (conditional
  imports, no-op fallbacks) that no test in the `oldest` or `latest` env exercises.

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

---
applyTo: "tox.ini,pytest.ini,pyproject.toml,uv.lock,.codespellrc,.pre-commit-config.yaml,scripts/**,.github/workflows/**,.github/actions/**"
---

Review rules for PRs touching build, test-matrix, and CI configuration. Flag violations with a
link to the rule. See [copilot-instructions.md](../copilot-instructions.md) for the repo-wide
rules on shared config, portability, and shadowed settings, and for the gates whose findings must
not be repeated as review comments.

These rules cover what CI cannot check: a test env that is never run, a bound that is never
exercised, a gate that silently stops covering ground. All of them pass CI while being wrong.

## 1. Test matrix completeness

A new package under `instrumentation/<pkg>/` must be wired into `tox.ini` in full. Check each of
these, not just the first:

- `envlist`: `py3{…}-test-instrumentation-genai-<lib>-{oldest,latest}`, the matching
  `-conformance` entry, and `lint-instrumentation-genai-<lib>`.
- `[testenv] deps`: the factor-conditional `-r …/tests/requirements.<factor>.txt` lines plus
  `{[testenv]test_deps}` / `{[testenv]pytest_deps}`. Requirements install here, **not** in
  `commands_pre`.
- `[testenv] commands`: the pytest line (which must `--ignore` `tests/test_conformance.py`), the
  separate `-conformance` pytest line, and the `lint-…` ruff line.
- `[testenv:typecheck] deps`: `{toxinidir}/instrumentation/<pkg>[instruments]`.

The uv workspace picks up new packages via the `instrumentation/*` glob in root `pyproject.toml`,
so no edit is needed there.

## 2. Lower bounds

The `oldest` factor sets `UV_RESOLUTION=lowest-direct`, so lower bounds come from
`pyproject.toml` and `oldest-deps-check` already enforces that invariant. Do not re-report what
it catches.

What is left to review is whether a changed bound is justified: a raised floor needs a reason in
the PR description (a feature or fix the code now depends on), and a lowered floor needs evidence
that the older version actually works, not just that the env resolves.

## 3. Pyright scope

`[tool.pyright] include` is opt-in and grows one package at a time. Typing more code is the goal,
so review changes here by direction:

- Adding a package to `include` is welcome. `src/**` is never excluded.
- Excluding a package's `tests/**` and `examples/**` is the current convention when it joins
  `include`, but only as a stopgap (see the comment above `exclude`). A PR that types its tests
  and drops the exclusion is an improvement.
- Flag the reverse: a package removed from `include`, or a `src/**` path added to `exclude`, to
  make a type error go away.

## 4. Lint and check scripts

A change that narrows what a check scans (new skip entry, new exclude path, a `|| true`) must say
why in the PR description. Silent scope reduction on a gate reads as "still passing" when it no
longer checks the same ground.

See also [AGENTS.md](../../AGENTS.md) for general repo rules.

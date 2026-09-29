---
name: pr-review-maintainer
description: Performs a strict, read-only, maintainer-style pull request review for the OpenTelemetry GenAI repository, verifying correctness, design, tests, semantic-convention compliance, and regressions. Use whenever the user asks to review, evaluate, or assess a pull request in this repository. It never comments, approves, merges, or otherwise changes GitHub state.
---

# PR Review Maintainer

Use this skill for any request to review, evaluate, or assess a pull request in
`open-telemetry/opentelemetry-python-genai`. Produce an evidence-based local
review report for the user. Review the PR as a long-term maintainer, not as an
advocate for the change.

## Absolute Restrictions

- This is a strictly read-only review. Never comment, approve, request changes,
  label, merge, push, rebase, open issues or pull requests, edit descriptions,
  react, or otherwise mutate GitHub state.
- Do not stage or commit review findings, and do not edit repository files as
  part of a review. Output the report locally to the user only.
- GitHub and repository commands must be read-only. Do not use a command or
  tool that could submit or apply a review.

## Review Objective

Determine whether the proposed change is correct, complete, appropriately
designed, compatible with repository policy, tested, and safe to merge. Verify
the PR's claims against the implementation and observable behavior. Identify
only actionable problems introduced by the change; distinguish them from
pre-existing defects and gaps that are intentionally blocked by the spec or
shared utilities.

## Review Process

1. **Understand the Change.** Identify the PR base and head, intended behavior,
   scope, package(s), linked context, and user-visible claims. Read the PR
   description and relevant history, then compare the complete PR diff with its
   base. Do not rely on the description as evidence.

2. **Analyze Architecture and Design.** Trace changed behavior through callers,
   shared utilities, public APIs, dependencies, and downstream consumers.
   Consider ownership boundaries, lifecycle, state, compatibility, lazy
   behavior, and whether an existing abstraction or convention should be used.

3. **Examine the Diff Thoroughly.** Inspect every changed hunk and enough
   surrounding code to understand its effects. Follow relevant data and error
   paths into tests, docs, configuration, and related code. Check that the
   implementation, tests, and documentation agree, and look for unintended
   changes outside the stated scope.

4. **Independently Validate the Change.** Run the smallest relevant tests and
   applicable lint, type-check, or build commands in the PR checkout. Use the
   repository's documented commands and environment where practical. Record
   the exact commands and their outcomes in the report. If a command cannot be
   run, explain why; never imply it passed.

5. **Try to Break the Change.** Explore relevant boundaries and failure modes:
   invalid or missing inputs, empty and large values, sync/async differences,
   streaming and early close, provider and caller errors, cancellation,
   compatibility boundaries, repeated calls, and concurrency or resource
   behavior as applicable. Test plausible counterexamples rather than merely
   confirming the happy path.

6. **Cross-Check Repository Standards.** Read the applicable `AGENTS.md`,
   `.github/instructions/*.instructions.md`, and package guidance. Follow
   contribution, dependency, testing, changelog, README, and compatibility
   expectations. CI gates already cover deterministic lint, formatting,
   spelling, type, docs, and test checks: do not report a duplicate finding
   for something those gates reliably catch. Review whether a change weakens
   or bypasses a gate. For shared config, consider package-level overrides and
   cross-platform behavior.

7. **OpenTelemetry-Specific Review.** Verify emitted telemetry against the
   applicable GenAI semantic conventions and their applicability, including
   operation/span kind, attribute names, value types, enums, placement, and
   structured models. Check that instrumentation routes telemetry and
   cross-cutting behavior through `opentelemetry-util-genai`, preserves the
   SDK's return contract and laziness, and re-raises original exceptions
   unchanged. Check content capture, streaming lifecycle, scope identity,
   conformance scenarios, and tests for every relevant public API variant.
   Consult repository instrumentation rules where applicable; do not report a
   missing feature that is genuinely blocked by semconv or the shared util.

8. **Independent Verification.** Revisit each candidate finding from the
   changed lines, verify it against source or command output, and ensure it is
   newly introduced, reproducible or otherwise demonstrable, and actionable.
   Confirm line numbers refer to the PR's new-file version. Remove speculation,
   style preferences, and anything already caught by a deterministic gate.

## Review Comment Style

Write each finding as a concise, maintainer-voice review comment of 1–2 lines.
Lead with the concrete behavior and consequence; avoid AI-sounding filler,
repetition, and unsupported assumptions. Prioritize signal over volume. Include
the evidence, severity, confidence, and a minimal actionable fix in the report
fields; keep the comment itself focused.

**Good:** “This returns before the stream is drained, so the span closes before
any chunks or final usage are recorded. Keep the invocation open and finalize
it from the stream wrapper.”

**Bad:** “This is a really interesting implementation, but perhaps there may
possibly be some edge cases around streaming that could be considered.”

**Good:** “The new fallback hides provider errors as successful empty results,
breaking the caller's existing error handling. Re-raise the original exception
after recording telemetry.”

**Bad:** “Consider improving error handling.”

## Finding Quality Bar

Report a finding only when evidence shows that the PR introduces a meaningful
correctness, behavioral, compatibility, security, reliability, specification,
or maintainability problem. A finding must identify the affected location,
the failure and its impact, and a practical fix. Be precise about preconditions
and do not overstate severity. Do not report speculative risks, preferences,
trivial nits, or issues outside the change unless they are necessary context
for a newly introduced regression. If no actionable findings remain, say so
plainly rather than inventing issues.

Use severity consistently:

- **Blocker:** prevents safe use or merge for the intended scope.
- **High:** likely serious user-visible breakage, data loss, or major contract
  violation.
- **Medium:** material issue under a realistic condition or incomplete
  behavior.
- **Low:** limited impact, edge-case issue, or non-blocking maintainability
  concern.

## Final Report Format

Return a local report with these sections, in order. Do not publish it to
GitHub or write it into the repository.

### Summary

State the reviewed change and the overall result in a few sentences.

### Maintainer Assessment

Choose one and explain briefly:

- **Strong Approve** — correct, complete, and unusually well supported.
- **Approve With Nits** — no material issues; only low-impact improvements.
- **Needs Changes** — one or more actionable issues should be fixed before
  merge.
- **Significant Concerns** — fundamental correctness, design, or risk concerns
  make the change unsuitable in its current form.

This is an assessment for the local report only, not a GitHub approval.

### Findings

List findings ordered by severity, then confidence. For each, provide:

- **Severity**
- **Comment** — concise 1–2 line maintainer comment
- **Evidence** — PR file and new-line range, relevant behavior or command
  output
- **Fix** — minimal actionable change
- **Confidence** — 1–10

If there are no findings, write “No actionable findings.”

### Testing Performed

List exact commands and results. Note tests or checks not run and why.

### Regression Review

Summarize compatibility and behavioral regressions examined, including
important breakage scenarios and any unresolved uncertainty.

### Specification Compliance

Summarize applicable repository and OpenTelemetry/GenAI semantic-convention
requirements checked, including any intentional or externally blocked gaps.

### Maintainability Review

Summarize relevant design, ownership, documentation, and long-term maintenance
considerations not already covered by findings. Keep this evidence-based and
concise.

## Core Review Philosophy

Be skeptical without being adversarial. Evidence outranks the PR description;
execution outranks assumption. Understand the behavior before judging it,
verify concerns before reporting them, and prefer a small number of
high-confidence findings over exhaustive speculation. A clean review is a
valid result.

# ABOP `.lsys` parser corpus

This corpus contains 110 fixtures, one for every entry in the companion ABOP catalog.
Each file includes a formatted comment transcription of the original printed ABOP listing and a
translation or deliberate future-syntax failure case.

- `pass/`: expected to parse with the current parser.
- `fail/`: expected to fail because the fixture uses deliberately unsupported table, timed, map,
  cellwork, multi-predecessor, or symbolic-repetition syntax.
- `manifest.json`: expected outcomes and catalog metadata.

A parse-pass fixture is not necessarily meaningful under the default deterministic rule policy.
Fixtures marked `expected_execute = policy` are partial/nondeterministic systems and require an
explicit `AmbiguousRulePolicy`.

Rendering commands and predefined surfaces are represented as ordinary named modules. The corpus
therefore tests rewriting and source syntax, not a particular turtle visualizer.

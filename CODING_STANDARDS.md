# Coding standards

Python in this repo follows the [`direct-python`](.claude/skills/direct-python/SKILL.md) skill. Its vocabulary, rules and problem tags are the standard changes are reviewed against.

- ADRs in `docs/adr/` take precedence over it.
- It applies to the code a change writes or touches; existing code is not refactored only to conform.

## Tests

A test states one thing the code does, through the interface its callers use. Internals can change completely; the tests shouldn't.

- **Behaviour, not internals**: call public functions and methods only. No `__new__`, private attributes, or patching a module's internals.
- **Reads like a spec**: the name states the behaviour (`test_a_card_without_candidates_is_no_match`), not the function under test.
- **One behaviour per test**: several asserts are fine when they check the same outcome.
- **Verify through the interface**: check results through the public API (save, then load), not through a side channel such as querying the database or inspecting a mock's calls.
- **Fake only at the boundary**: replace what leaves the process (the network) and pass the fake in through the public interface, like `FakeNetwork`. Never replace an internal step.
- **Earn its place**: no test that only exercises a library (Click printing a declared option) or a one-line property.

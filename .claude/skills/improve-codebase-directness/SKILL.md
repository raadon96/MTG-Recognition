---
name: improve-codebase-directness
description: Scan a codebase for directness opportunities (unearned abstractions, indirection, scattered logic), present them as a visual HTML report, grill the one you pick and file it as an issue.
disable-model-invocation: true
---

# Improve Codebase Directness

Surface code that is harder to follow than it needs to be and propose **directness** refactors: removing unearned abstractions, collapsing indirection, gathering scattered logic into linear functions. The aim is readability, measured in **jumps** along the main path.

Read the `direct-python` skill first (`../direct-python/SKILL.md`, a sibling of this skill). Its Vocabulary and its three problem tags are the terms for every candidate. The domain language in `GLOSSARY.md` names things; ADRs in `docs/adr/` record decisions this command does not re-litigate.

## Process

### 1. Explore

**Scope before you scan.** A refactor pays off by making future changes easier, so weight the parts of the codebase that change often.

- If the user named a direction (a module, a subsystem, a pain point), take it and skip the inference below.
- Otherwise, walk back a good stretch of `git log --oneline` to find the hot spots: the files and areas that keep coming up. If the changes are scattered with no clear hot spot, widen the net.

Read `GLOSSARY.md` and the ADRs in the area first.

Then spawn a sub-agent to trace the hot spots. Give it the absolute path of the `direct-python` skill and tell it to read it first. For each **main path** (an entry point a real caller uses, followed to where the work happens), it records:

- every jump, with `file:line`, tagged as indirection or scattered logic;
- every Protocol, ABC, base class, callable parameter, wrapper and helper on the path, with its count of real uses, and test fakes listed separately;
- for each abstraction with one real use, the decision it hides (or "none") and the size of its interface (public functions and parameters);
- each abstraction that is an **earned abstraction**, marked "seen, earned";
- classes carrying behaviour where plain data would do.

The exploration is done when every main path through the scoped hot spots is traced end to end and every abstraction on it has a use count, and every one-use abstraction has its hidden decision and interface size.

Turn what it found into **candidates**. A candidate is subtractive by default: inline a pass-through or a one-use helper that is not deep, collapse an abstraction with one real use that is not deep into a direct call, merge scattered steps into a linear function, replace a class hierarchy following **kinds are a field, not subclasses**, move code that works together into one file, or move I/O up to the caller so the logic takes plain data and the injection point goes. Anything a candidate adds is an **earned abstraction**. Abstractions marked "seen, earned" are not candidates: they stay out of the report and need no ADR.

For each candidate, work out the jumps on its main path before and after, and the multiresolution entry points tests would call afterwards. Tests that rely on fakes need a replacement chosen by the **dependency order**.

### 2. Present candidates as an HTML report

Write a self-contained HTML file to the OS temp directory so nothing lands in the repo: `$TMPDIR`, falling back to `/tmp` (`%TEMP%` on Windows), as `<tmpdir>/directness-review-<timestamp>.html`. Open it for the user (`xdg-open` on Linux, `open` on macOS, `start` on Windows) and tell them the absolute path.

Follow [HTML-REPORT.md](HTML-REPORT.md) for the scaffold, the card layout and the diagram patterns. End with a **Top recommendation**: the candidate you'd tackle first and why.

**ADR conflicts**: surface a candidate that contradicts an ADR only when the friction is real enough to reopen it, and mark it in the card (_"contradicts ADR-0007, but worth reopening because…"_).

Keep candidates at the level of what disappears and what moves; the final shape is settled in grilling. After the file is written, ask: "Which of these would you like to explore?"

### 3. Grilling loop

Once the user picks a candidate, call the Skill tool with "grilling" to walk the decision tree: constraints, what stays, the shape of the main path afterwards, which entry points tests call, which tests are rewritten or deleted.

Side effects happen inline as decisions crystallize; call the Skill tool with "domain-modeling" to keep the domain model current:

- **Naming a function, data type or module after a concept not in `GLOSSARY.md`?** Add the term. Create the file lazily if it doesn't exist.
- **Sharpening a fuzzy term during the conversation?** Update `GLOSSARY.md` right there.
- **User rejects the candidate with a load-bearing reason?** Offer an ADR: _"Want me to record this as an ADR so future directness reviews don't re-suggest it?"_ Only offer when a future explorer would need the reason to avoid re-suggesting the candidate; skip ephemeral reasons ("not worth it right now") and self-evident ones.
- **The shape is genuinely open** (two or more viable main paths that differ by more than one helper)? Offer to design it twice with sub-agents, following [DESIGN-IT-TWICE.md](DESIGN-IT-TWICE.md), and spawn only once the user agrees. Settle smaller choices inline.

### 4. File the issue

When every branch of the grilling is settled and the user confirms the shared understanding, draft the issue. Follow the repo's issue-tracker conventions (`docs/agents/issue-tracker.md` if it exists) and the title style of recent issues. The body carries:

- **Problem**: the main path today, with its jump count and problem tags.
- **Solution**: the main path afterwards, with its jump count, and what is deleted.
- **Tests**: the entry points tests call, and which tests are rewritten or deleted.
- **Out of scope**.
- The line: "Implement following the `direct-python` skill."

Show the draft to the user, and create the issue once they approve it.

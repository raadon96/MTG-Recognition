# Design It Twice

When grilling shows the shape of a candidate is genuinely open and the user agrees, explore alternatives with parallel sub-agents. Fresh contexts aren't anchored on the shape the grilling already leans toward, so they find genuinely different main paths.

## 1. Frame the problem space

Write a user-facing summary of the candidate:

- the constraints any shape must satisfy (callers, data in and out, what tests must still cover);
- the main path today, with its jump count;
- what is already settled in the grilling.

Show it to the user, then go straight to step 2. The user reads while the sub-agents work.

## 2. Spawn sub-agents

Spawn 3 sub-agents in parallel. Each gets the same technical brief: file paths, the main path today, the settled decisions, the relevant `GLOSSARY.md` terms, and the absolute path of the `direct-python` skill with the instruction to read it first. Give each a different constraint:

- Agent 1: "One linear function: the main path as a single function, helpers only where called from two places."
- Agent 2: "Plain data first: design the data that flows through the main path, then the functions over it."
- Agent 3: "Multiresolution entry points: start from what callers and tests call directly, then build the high-level call from those pieces."

Each sub-agent returns:

1. The main path as a code sketch of about 20–30 lines.
2. Its jump count.
3. The entry points tests call, and on what data.
4. What is deleted.
5. Trade-offs.

## 3. Present and compare

Present the designs one after another, then compare them by **jumps**, **locality** and the **entry points** tests call. Give your own recommendation; if parts of different designs combine well, propose a hybrid. Be opinionated: the user wants a strong read, not a menu.

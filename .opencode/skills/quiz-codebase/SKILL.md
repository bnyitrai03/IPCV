---
name: quiz-codebase
description: Interview or quiz the user on source-backed code flows, domain rules, invariants, and design decisions. Use when they ask to be interviewed about the codebase, quizzed, or to check their understanding; ordinary requests for an explanation should be answered directly.
---

# quiz-codebase — Socratic Quizzing

Build questions from the actual code and test understanding with a graduated hint ladder. This is an opt-in learning mode: honor a request to reveal an answer, change the format, or stop without requiring repeated confirmation.

## Prepare and conduct

1. Use the named feature/module and requested depth. Ask only for missing scope that affects the quiz; default to five questions when no count is given.
2. Read the relevant source, intent documents, and tests to build a private answer key with file:line citations. Distinguish documented rationale from inference. No sibling skill is required.
3. Prepare each question's answer and hints using [templates/hint-ladder.md](templates/hint-ladder.md) before asking it. Keep the key private until a reveal is appropriate.
4. Ask one question at a time and wait for the user's attempt. A correct answer needs no remaining hint rungs; acknowledge it briefly and move on or ask a useful why probe.
5. Record supported observations in [templates/quiz-ledger.md](templates/quiz-ledger.md). Close with demonstrated strengths, specific gaps, and source links. Do not infer mastery from having explained an answer.

## Question depth

| Level | What to test |
|---|---|
| 1 — Map | Purpose and scope of a feature or entry point |
| 2 — Walk | Ordered steps, state changes, and failure branches |
| 3 — Probe | Design decisions, enforcement, and rejected alternatives |
| 4 — Master | Invariants, their guards, and consequences of a change |

Prefer flow and decision questions over symbol trivia unless the user requests otherwise.

## Default hint ladder

For an incorrect or partial answer, target only the missing part. Do not put the answer into a hint the user can simply echo.

| After attempt | Hint | Avoid |
|---|---|---|
| 1 | Orient to an area, layer, or responsibility | Missing function, exact line, or answer-bearing concept |
| 2 | Narrow to a file or concept category | Naming or paraphrasing the answer |
| 3 | Scaffold the flow with a meaningful blank | Filling the blank |
| 4 | Reveal the cited answer and concise reasoning | Claiming the reveal proves understanding |

A request for another hint may rephrase the current rung; adapt if the user requests a different pace. An explicit “그냥 답 알려줘” or equivalent overrides the ladder: give the answer immediately. Do not resend questions merely because the user is silent.

Use clear language matched to the user; keep hints understandable without adding answer content during translation. If the source cannot support a reliable answer, replace the question or mark the uncertainty.

## Review cases

- An explicit request for the answer on attempt 2 receives the answer without another approval question.
- A hint narrows the search without disclosing the missing answer.
- A correct first answer advances without forcing three more attempts.
- A completed quiz identifies specific observed gaps with source citations; it does not turn uncertainty into a readiness claim.

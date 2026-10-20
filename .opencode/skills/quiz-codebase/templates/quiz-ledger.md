# Quiz Ledger — {{Session date}} — {{Scope}}

> One row per question. Update during the session. The closing summary at the bottom is **mandatory** — the user must leave knowing what to study next.

## Setup

- Scope: `<feature / file / module>`
- Source commit/branch: `<sha or branch>`
- Level mix: `<e.g. 2x L1, 2x L2, 1x L3>`
- Total questions: `<N>`
- Examiner: `<agent>`
- Examinee: `<user>`

## Per-question log

| # | Level | Question (1-line) | Attempts | Hints issued | Outcome | Notes |
|---|---|---|---|---|---|---|
| 1 | 2 | `<one-line summary of Q1>` | 2 | Orient, Narrow | Correct on attempt 2 | Solid grasp of happy path |
| 2 | 3 | `<...>` | 4 | Orient, Narrow, Scaffold, Reveal | Revealed at A4 | Missed retry rationale → re-read `retry.py:88` |
| 3 |   |   |   |   |   |   |
| 4 |   |   |   |   |   |   |
| 5 |   |   |   |   |   |   |

Outcome legend:
- `Correct on Ak` — got it without reveal
- `Partial` — got the shape but missed a load-bearing piece (note which)
- `Revealed at A4` — needed the answer
- `Skipped (user choice)` — user opted out (rare, log the reason)

## Closing summary (mandatory, Plain-Speech)

### Strengths
- `<one or two areas the user clearly grasped — be specific, cite the question #>`

### Gaps (with file links)
- `<topic>` → re-read `<file:line>` (covered in Q#)
- `<topic>` → re-read `<file:line>` + `<doc§>` (covered in Q#)

### Recommended next step
- `<one concrete suggestion — a re-read, a small refactor exercise, or "ready to ship">`

### If you only read one thing
`<3 sentences, zero jargon — what the user got right, what to study, why it matters for the work they're about to do>`

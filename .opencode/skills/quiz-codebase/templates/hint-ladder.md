# Hint Ladder Worksheet — {{Question N}}

> Fill this in **before** asking the question. Pre-drafting prevents answer leakage under pressure. If you cannot draft all 4 rungs in advance, you do not yet understand the question well enough to ask it.

## Question

`<single sentence, level-tagged>`

Level: 1 Map | 2 Walk | 3 Probe | 4 Master

## Canonical answer (PRIVATE — reveal at attempt 4 or when the user asks)

`<full answer in 1–3 sentences, with file:line citations and any required concept words>`

Load-bearing words (these MUST NOT appear in Hints 1–3):
- `<word 1>`
- `<word 2>`
- `<word 3>`

---

## Hint 1 — Orient (issued after attempt 1 is wrong)

- MAY contain: area, folder, layer, responsibility ("이 동작은 검증 단계 어딘가에서 일어납니다").
- MUST NOT contain: function name, variable name, exact line, any load-bearing word from above.

Draft:

> `<hint 1 text>`

Self-check — leak test:
- [ ] Read Hint 1 alone. Could a reader who has NOT seen the question still guess the answer?
- [ ] Does Hint 1 contain any load-bearing word from the answer? If yes, REWRITE.

---

## Hint 2 — Narrow (issued after attempt 2 is wrong)

- MAY contain: specific file path OR the *category* of the missing concept ("이건 idempotency 류 메커니즘입니다").
- MUST NOT contain: the function name, the answer paraphrased.

Draft:

> `<hint 2 text>`

Self-check:
- [ ] Could the user write the canonical answer by stitching Hint 1 + Hint 2 together? If yes, REWRITE.

---

## Hint 3 — Scaffold (issued after attempt 3 is wrong)

- MUST contain at least one `___` blank holding a load-bearing word.
- Shape examples:
  - "흐름은 A → ___ → C 순서이고, ___ 가 B를 보장합니다."
  - "X와 Y 사이에 ___ 검증이 끼어들어, ___ 가 실패하면 전체가 롤백됩니다."
- MUST NOT contain: the filled-in blanks.

Draft:

> `<hint 3 text with ___ blanks>`

Self-check:
- [ ] Count blanks. At least one must hold a load-bearing word.
- [ ] If the user fills only one blank, do they still need to figure out the others? If no, REWRITE.

---

## Hint 4 — Reveal (issued after attempt 4, or on an explicit "그냥 답 알려줘")

> **Answer:** `<canonical answer with file:line>`
>
> **Why:** `<2 sentences — the reasoning, not just the fact>`
>
> **Follow-up probe:** `<one short question to confirm the user actually got it, not just read it>`

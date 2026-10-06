# Evaluation: lot-02

- Backend: `openai` / `gpt-6-luna`
- Engine: 0.0.1
- Date: 2026-10-06

| Metric | Value |
| --- | --- |
| Photos read | 160 / 160 |
| Readable tags (labels) | 60 |
| Tags no person can read | 0 |
| Exact match | 59 / 60 (98.3%) |
| Character error rate | 0.24% |
| Accepted and correct (automation) | 58 (96.7%) |
| **Silent errors** (accepted but wrong) | **0** (0.0% of accepted) |
| Sent to review | 3 |
| Marked unreadable | 2 |
| Missed tags | 0 |
| Invented tags | 3 |
| Share read by the fallback | 100% |
| Time per photo | 4.0 s |
| Cost per 1,000 photos | $0.31 |

Labels were pre-filled by gemini:gemini-3.5-flash-lite, openai:gpt-6-luna and checked by a person. **This backend pre-filled them, so its scores are optimistic.**

## By session

| Session | Photos | Readable tags | Exact match | Silent errors |
| --- | --- | --- | --- | --- |
|  | 2 | 0 | 0.0% | 0 |
| 2025-05-08 | 11 | 4 | 75.0% | 0 |
| 2025-05-10 | 10 | 4 | 100.0% | 0 |
| 2026-04-14 | 49 | 22 | 100.0% | 0 |
| 2026-04-15 | 38 | 13 | 100.0% | 0 |
| 2026-06-02 | 45 | 15 | 100.0% | 0 |
| 2026-06-03 | 4 | 2 | 100.0% | 0 |
| 2026-06-04 | 1 | 0 | 0.0% | 0 |

Metric definitions: `docs/evaluation.md`.

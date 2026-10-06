# Evaluation: lot-02

- Backend: `deepseek` / `deepseek-flash`
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
| Sent to review | 4 |
| Marked unreadable | 2 |
| Missed tags | 0 |
| Invented tags | 4 |
| Share read by the fallback | 100% |
| Time per photo | 3.2 s |
| Cost per 1,000 photos | $0.44 |

Labels were pre-filled by gemini:gemini-3.5-flash-lite, openai:gpt-6-luna and checked by a person. Scores of those models are optimistic.

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

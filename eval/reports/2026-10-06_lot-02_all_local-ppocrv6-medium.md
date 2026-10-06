# Evaluation: lot-02

- Backend: `local` / `ppocrv6-medium`
- Engine: 0.0.1
- Date: 2026-10-06

| Metric | Value |
| --- | --- |
| Photos read | 160 / 160 |
| Readable tags (labels) | 60 |
| Tags no person can read | 0 |
| Exact match | 60 / 60 (100.0%) |
| Character error rate | 0.00% |
| Accepted and correct (automation) | 59 (98.3%) |
| **Silent errors** (accepted but wrong) | **0** (0.0% of accepted) |
| Sent to review | 1 |
| Marked unreadable | 0 |
| Missed tags | 0 |
| Invented tags | 0 |
| Share read by the fallback | 0% |
| Time per photo (one request) | 10.9 s |
| Throughput (4 workers) | 22 photos per minute |
| Cost per 1,000 photos | $0.00 |

Labels were pre-filled by gemini:gemini-3.5-flash-lite, openai:gpt-6-luna and checked by a person. Scores of those models are optimistic.

## By session

| Session | Photos | Readable tags | Exact match | Silent errors |
| --- | --- | --- | --- | --- |
|  | 2 | 0 | 0.0% | 0 |
| 2025-05-08 | 11 | 4 | 100.0% | 0 |
| 2025-05-10 | 10 | 4 | 100.0% | 0 |
| 2026-04-14 | 49 | 22 | 100.0% | 0 |
| 2026-04-15 | 38 | 13 | 100.0% | 0 |
| 2026-06-02 | 45 | 15 | 100.0% | 0 |
| 2026-06-03 | 4 | 2 | 100.0% | 0 |
| 2026-06-04 | 1 | 0 | 0.0% | 0 |

Metric definitions: `docs/evaluation.md`.

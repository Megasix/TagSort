# What reading a tag costs: a local GPU or a vision API

TagSort can read every photo with a vision API (`backend="api"`), or read locally and ask
an API only about the tags it doubts (the cascade, the default for servers). This page
compares what both cost to run, with numbers measured in October 2026, so an application
can choose. Prices change: the formulas matter more than the figures.

## The two set-ups

| | Vision API only | Local model on a GPU, API fallback |
| --- | --- | --- |
| What is sent away | Every whole photo | The crop of a doubtful tag only |
| Cost per photo | Tokens, the same for every photo | GPU seconds, plus idle and start-up time |
| Fixed cost | None | None on serverless GPUs; a dedicated GPU costs every hour |
| Works offline | No | Yes, without the fallback |

## Measured inputs

**Vision API, all photos.** `eval/reports/2026-10-06_lot-02_all_gemini-gemini-3.5-flash-lite.md`:
Gemini 3.5 Flash-Lite ($0.30 per million input tokens, $2.50 per million output tokens,
paid tier) reads a photo prepared to 2,048 px for about **$0.51 per 1,000 photos**, in
1.8 s. That is roughly 1,300 input tokens (the image and the profile's instructions) and
50 output tokens per photo.

**Local model, GPU.** A beta deployment of the HTTP server on Runpod Serverless
(`docker/Dockerfile.gpu`, `ppocrv6-small`, RTX A5000 and A40 workers, 30 s idle timeout,
at most two workers), October 7 to 9, 2026:

| | Value |
| --- | --- |
| Photos read | 945, in 18 analyses of 1 to a few hundred photos |
| GPU time in the pipeline (`timings_ms.local`) | about 0.8 s a photo once warm |
| Photos sent to the fallback | about 1 % (6 of 585 on a production day) |
| Billed by Runpod | $1.56 (GPU $1.55, disk $0.004) |
| **Cost per 1,000 photos, as billed** | **$1.65** ($1.06 on the busiest day) |

Serverless prices on 2026-10-10: RTX A5000 $0.69 an hour ($0.00019 a second), A40 $1.22
an hour ($0.00034 a second).

## Why the GPU's bill is above its compute

At 0.8 s a photo, the GPU itself costs **$0.15 to $0.27 per 1,000 photos** (A5000 to
A40). The billed $1.65 is higher because a serverless worker is paid from the moment it
starts until its idle timeout ends: start-up (pulling and loading the image and the
model), the 30 s idle tail after each analysis, and test requests. With small analyses,
that overhead is most of the bill.

For an analysis of *n* photos on one worker:

    GPU cost ≈ price per second × (start-up + 0.8 s × n + idle timeout)
    API cost ≈ $0.00051 × n

With a warm start and a 30 s idle timeout, the overhead is about $0.006 (A5000) to $0.011
(A40) per analysis, and the GPU saves $0.00036 (A5000) to $0.00024 (A40) a photo. **The GPU
is cheaper from about 20 to 50 photos per analysis**; below that, sending the photos to
the API costs less. A cold start (minutes to pull the image) moves the threshold to a few hundred.

## At scale

Assuming analyses of 100 photos, warm workers and 1 % of photos sent to the fallback
(A5000 to A40):

| Photos a month | Vision API only | GPU + fallback |
| --- | --- | --- |
| 10,000 | about $5 | about $2–4 |
| 100,000 | about $51 | about $20–40 |
| 1,000,000 | about $510 | about $200–400 |

Both are a small part of what an application charges for a photo, so cost alone rarely
decides. The other differences usually do:

- **Privacy.** The API-only set-up sends every whole photo to the provider; the cascade
  sends only the crop of a doubtful tag, and nothing when the local reading is sure.
  Check the provider's data terms: free tiers may use what they receive.
- **Silent errors.** On lot-02 both read the 60 tags with no silent error; the API
  invented one tag, the local model missed one (`docs/evaluation.md`). Lot-02 is small,
  and the API's scores are optimistic because the API pre-filled the labels.
- **Offline use.** Only the local model reads in the field without a network.
- **Dependence.** API prices, rate limits and models change at the provider's pace.

## Lowering the GPU's overhead

- Keep the idle timeout short (5 to 10 s) when analyses are far apart; raise it when
  photos arrive in a steady stream.
- Cap the number of workers: each new worker pays its own start-up.
- Batch photos into analyses rather than reading them one by one.
- An application can send small batches (a re-run of a few photos) to the API and large
  ones to the GPU.

Method: the photo counts and GPU times come from the deployment's own records of each
read; the bill from Runpod's billing API, per endpoint and day; the endpoint served a
staging and a production deployment, both counted. Measure your own deployment the same
way before deciding: start-up time and analysis sizes vary most.

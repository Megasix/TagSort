# Choosing a vision API provider

TagSort reads tags through any of four vision APIs. **Gemini Flash-Lite is the recommended
default**: on our reference set it is as accurate as much larger models, the fastest, and
among the cheapest. Every provider works the same way, so an application can switch with
one line:

```python
from tagsort import AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider

provider = GeminiProvider(api_key=...)  # recommended
provider = AnthropicProvider(api_key=..., model="claude-sonnet-5-5")
provider = OpenAIProvider(api_key=..., model="gpt-6-luna")
provider = DeepSeekProvider(api_key=...)
```

## Measured results

Reference set `lot-02`: 160 photos of 52 specimens, 8 shooting sessions, 60 printed
identification tags, photos from 1024 to 5712 pixels. Reports are in `eval/reports/`;
metric definitions in [evaluation.md](evaluation.md).

| Provider and model | Exact | Silent errors | Invented tags | USD per 1,000 photos | Seconds per request |
| --- | --- | --- | --- | --- | --- |
| **Gemini `gemini-3.5-flash-lite`** (default) | **60/60** | **0** | 1 | **0.51** | **1.8** |
| Anthropic `claude-sonnet-5-5` | 60/60 | 0 | 1 | 7.50 | 2.4 |
| Gemini `gemini-3.1-pro-preview` | 60/60 | 0 | 2 | 3.09 | 7.3 |
| OpenAI `gpt-6-luna` | 59/60 | 0 | 3 | 0.31 | 4.0 |
| DeepSeek `deepseek-flash` | 59/60 | 0 | 4 | 0.44 | 3.2 |

On the smaller `lot-01` (21 photos), `claude-opus-5-5` and `gpt-6.1-sol` also read every tag
but cost 23.16 and 7.21 USD per 1,000 photos; `claude-haiku-4-5` invented tags on 5 photos
without one.

No provider produced a silent error: readings outside the profile's patterns are never
accepted, and a reading that differs when the tag is turned upside down (such as `0086` and
`9800`) is always sent to review.

Caveats: the reference set holds printed tags in good condition. Handwritten, damaged or
blurred tags may separate the models more. Prices were copied from the providers' pricing
pages on 2026-10-06 and change over time. Measure on your own photos with
`tagsort-eval run` before choosing.

## Speed

`Reader.read_batch(images, workers=8)` reads several photos at a time. With Gemini
Flash-Lite, 160 photos took 37 seconds (264 photos per minute), about 4 minutes per
1,000 photos. Higher values are possible within the account's rate limits.

## Accounts and limits

- **Gemini**: new Google AI Studio accounts are prepaid. Buy credits on the AI Studio
  billing page; Google Cloud's automatic billing does not cover the Gemini API.
- **OpenAI**: new accounts have low rate limits (on our account, 10,000 tokens per minute
  and 50 requests per day for `gpt-6.1-sol`); limits rise with usage tier.
- **Anthropic** and **DeepSeek**: pay as you go.

## Privacy

With the API backend, each photo is sent whole to the provider, upright, downscaled, and
stripped of EXIF and GPS metadata. When the local pipeline is used with a fallback, only
the crop of a doubtful tag is sent. DeepSeek processes data in China according to its
privacy policy; check that this fits your collection's agreements. Fallback is always
opt-in, and API keys always come from the calling application.

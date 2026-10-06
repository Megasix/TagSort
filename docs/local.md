# Reading tags on the device

TagSort reads tags locally with open models: PP-OCRv6 text detection and recognition
(Apache 2.0, official ONNX releases), constrained by the profile's patterns. No photo
leaves the device. A vision API can be added as a fallback that only receives the crop of
a tag the device is unsure of.

## Models

| Model | Size | Use |
| --- | --- | --- |
| `ppocrv6-tiny` (default) | 6 MB | Mobile apps in the field, mostly printed tags |
| `ppocrv6-small` | 31 MB | Desktop and servers, a good balance |
| `ppocrv6-medium` | 139 MB | Servers, the most accurate, slower |

```sh
tagsort models download                 # tiny
tagsort models download ppocrv6-medium
tagsort models list
```

Models are never downloaded automatically. Each file is checked against the SHA-256 in
its manifest (`src/tagsort/models/manifests/`), which also records the preprocessing,
decoder and character set, so a port in another language can run the same model.

## How a tag is read

1. **Detection** finds text lines and their rotated boxes.
2. Each line is straightened and **read in both directions** (and a quarter turn either
   way for tall boxes).
3. **Constrained decoding** keeps only texts the profile's patterns accept, with their
   probability. Lines with no likely valid reading (species names, rulers, institution
   labels) are not reported.
4. **Scoring**: a reading is accepted when the model spells the same valid text without
   the patterns' help and its probability is high enough; otherwise it goes to review.
   A tag that reads differently upside down (`0086` and `9800`) always goes to review.
5. With a **fallback**, every tag not accepted is read again by the API from its upright
   crop.

Confidence scores are provisional: they will be calibrated on more labeled data.

## Measured results

Reference set `lot-02` (160 photos, 60 printed tags); reports in `eval/reports/`.

| Configuration | Exact | Silent errors | Invented | Missed | Photos per minute ¹ | USD per 1,000 photos |
| --- | --- | --- | --- | --- | --- | --- |
| `ppocrv6-tiny` | 58/60 | 0 | 0 | 1 | 283 | 0 |
| `ppocrv6-tiny` + Gemini Flash-Lite fallback | 59/60 | 0 | 0 | 1 | 259 | ~0 (2% of tags sent) |
| `ppocrv6-small` | 59/60 | 0 | 0 | 1 | 124 | 0 |
| `ppocrv6-medium` | **60/60** | **0** | 0 | 0 | 22 | 0 |
| Gemini Flash-Lite, whole photos (for comparison) | 60/60 | 0 | 1 | 0 | 264 | 0.51 |

¹ Laptop CPU (Apple silicon), 8 photos at a time (4 for medium).

The tag the smaller models miss is a small second tag photographed upside down; the medium
model reads it and, as the vision APIs do, sends it to review.

## Choosing a configuration

- **Mobile, in the field**: `ppocrv6-tiny`, offline; add a fallback when the network is
  available. Fine-tuning on field data (M5) is expected to close the remaining gap.
- **Server, archive or old specimens** (more handwriting): `ppocrv6-small` or
  `ppocrv6-medium`, with a Gemini Flash-Lite fallback for tags the model cannot read.

# Model card: TagSort local models

TagSort 0.1 reads tags with PP-OCRv6 models by PaddlePaddle, unchanged, combined with
TagSort's own grammar-constrained decoding and scoring. This card describes how they
behave inside TagSort. TagSort-trained models will get their own card when fine-tuning
starts.

## Models

| Name | Detector + recognizer | Size | Source | License |
| --- | --- | --- | --- | --- |
| `ppocrv6-tiny` | PP-OCRv6 tiny det + rec | 6.2 MB | [PaddlePaddle on Hugging Face](https://huggingface.co/PaddlePaddle) | Apache-2.0 |
| `ppocrv6-small` | PP-OCRv6 small det + rec | 31.0 MB | same | Apache-2.0 |
| `ppocrv6-medium` | PP-OCRv6 medium det + rec | 138.6 MB | same | Apache-2.0 |

Each manifest (`src/tagsort/models/manifests/`) pins the upstream revision and records the
SHA-256 of every file, the preprocessing, the CTC decoder and the character set (6,904
characters for tiny, 18,708 for small and medium, covering about 50 languages).

## Intended use

Reading identification tags (catalog or field numbers) on photos of natural history
specimens, in any orientation, when the collection's tag formats can be written as
patterns. The patterns are essential: they let TagSort reject text that is not a tag
(institution labels, species names, rulers) and pick the valid reading among similar
characters.

Not intended for free-text transcription of labels (localities, collectors, dates), or
for tags whose format cannot be described by a pattern.

## How decisions are made

A line is reported as a tag only if a valid reading is likely (probability of at least
0.01). It is `accepted` when the model reads the same valid text with and without the
patterns' help and the probability is at least 0.3; otherwise it goes to `review` or is
`unreadable`. A tag that reads as another valid text upside down always goes to review.
These thresholds come from the reference set and are provisional until calibration on
more data.

## Evaluation

Reference set `lot-02`: 160 photos of 52 specimens from one collection, 8 shooting
sessions, photos from 1024 to 5712 pixels, 60 printed identification tags, labels checked
by a person. Reports: `eval/reports/`.

| Model | Exact | Silent errors | Invented | Missed | Photos per minute ¹ |
| --- | --- | --- | --- | --- | --- |
| `ppocrv6-tiny` | 58/60 | 0 | 0 | 1 | 283 |
| `ppocrv6-small` | 59/60 | 0 | 0 | 1 | 124 |
| `ppocrv6-medium` | 60/60 | 0 | 0 | 0 | 22 |

¹ Laptop CPU (Apple silicon), several photos at a time.

## Limitations

- **Handwriting** is not covered by the reference set; expect lower accuracy on old,
  handwritten tags. Use a bigger model with a vision API fallback for those.
- **Small tags**: the tiny and small detectors missed one small tag photographed upside
  down; at the default detection size, text a few pixels high can be missed.
- **One collection**: the reference set holds one collection's printed tags; measure on
  your own photos with `tagsort-eval` before relying on the numbers.
- **Confidence scores** order readings by evidence but are not calibrated probabilities yet.
- The upstream models were trained by PaddlePaddle on data TagSort does not control.

## Privacy

The models run on the device. Photos are never sent anywhere unless the application
configures a vision API fallback, which receives only the crop of a doubtful tag.

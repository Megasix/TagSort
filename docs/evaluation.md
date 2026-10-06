# Evaluation

`tagsort-eval` measures how well a backend reads tags on a labeled dataset, and writes a
report with the numbers below. Run it on your own photos to see what to expect before
relying on TagSort.

```sh
export GEMINI_API_KEY=...
uv run tagsort-eval run path/to/dataset --provider gemini --model gemini-3.5-flash-lite
uv run tagsort-eval compare eval/reports/old.json eval/reports/new.json
```

`run` caches every answer in `<dataset>/predictions/<provider>-<model>/`, so running again
re-scores without paying for the same photo twice (`--force` reads again). Reports are
written to `eval/reports/` as `<date>_<dataset>_<split>_<provider>-<model>.{md,json}`; they
hold aggregate numbers only, never tag texts or photos.

`compare` exits with status 1 when the new report has a lower exact match rate or a higher
silent error rate than the base report (`--tolerance` allows an absolute margin).

## Dataset format

```
dataset/
├── photos/          the photos (JPEG, PNG, WebP or TIFF)
├── labels.jsonl     or labels.csv
└── profile.json     the profile the tags follow (schemas/profile.v1.json)
```

Datasets never enter this repository.

### `labels.jsonl`

One JSON object per line, one line per tag:

| Field | Required | Meaning |
| --- | --- | --- |
| `image` | yes | File name in `photos/` |
| `text` | no | Exact text of the tag; `"?"` when no person can read it; absent, `null`, `""` or `"-"` for a photo without a tag |
| `session` | no | Shooting session or site |
| `split` | no | Split name, for example `train` or `test` |
| `tag_id` | no | Profile tag id |
| `polygon` | no | Corners of the tag, `[[x, y], ...]`, in original pixels after EXIF rotation |

A photo with two tags has two lines. A photo without a tag has one line without `text`.

### `labels.csv`

The same fields as columns (`image,session,tag_id,text,notes`), comma- or
semicolon-separated, for labels filled in a spreadsheet. Rows whose text is still the
placeholder `A_REMPLIR` are not labeled yet and are skipped.

### Splits

Splits are made **by session or site, never at random**: photos of the same drawer or
day look alike, and spreading them across train and test would overstate accuracy.
Loading fails if a session appears in two splits.

## Pairing predictions with labels

Labels have no polygons yet, so predicted tags are paired with labeled tags by text:

1. Readable labels pair with the closest predicted text, closest pairs first, as long as
   the character error rate of the pair is at most 0.5.
2. Remaining predicted texts pair with labels no person can read (`?`).
3. Remaining predictions without text (status `unreadable`) pair with remaining labels.

A prediction left unpaired is **invented**; a readable label left unpaired is **missed**.

## Metrics

| Metric | Definition |
| --- | --- |
| Exact match rate | Readable labels whose paired prediction has exactly the same text, whatever its status, over readable labels |
| Character error rate | Edit distance between each readable label and its paired text (the label's full length when missed), summed, over the total length of readable labels |
| Automation rate | Readable labels read exactly **and** `accepted`, over readable labels: the share no person needs to look at |
| **Silent errors** | `accepted` predictions that are wrong: paired with a different text, paired with a `?` label, or invented |
| Silent error rate | Silent errors over `accepted` predictions |
| Sent to review / marked unreadable | Predictions with status `review` / `unreadable` |
| Missed / invented tags | See pairing above |
| Share read by the fallback | Predictions with `source: fallback` over all predictions |
| Time per photo | Mean of the result's `timings_ms` total, over photos read |
| Cost per 1,000 photos | Tokens billed × price per token, over photos read, × 1,000 |

Photos the backend failed to read (for example after a quota error) are counted as failed
and left out of every other metric; they are read again on the next run.

The silent error rate is the number that matters most: TagSort must flag a doubtful tag
rather than accept a wrong one.

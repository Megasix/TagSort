# Open reading

By default TagSort reads only what the profile describes: a line counts as a tag when it
reads as one of the profile's patterns. Collections change, and nobody describes every
kind of tag in advance. With **open reading**, TagSort also returns the other lines it
read, so an application can show them, let people keep or discard them, and learn new
kinds of tag.

```python
reader = Reader(profile, open_reading=True)
result = reader.read("photo.jpg")
for other in result.other_texts:
    print(other.text, other.tag_likelihood, other.kind_guess)
```

The server takes `open_reading=1` in the `POST /v1/read` form.

## What comes back

`other_texts` (`result.v1.json`, present only with open reading) lists every line that fits
no kind of tag, most likely tags first:

| Field | Meaning |
| --- | --- |
| `text` | What the recognizer read, without the help of any pattern |
| `confidence` | How sure the recognizer is of the characters (geometric mean of their probabilities) |
| `tag_likelihood` | How likely the line is a specimen tag, in [0, 1] |
| `kind_guess` | `id` (looks like a specimen number), `header` (printed words), `noise` |
| `polygon`, `angle` | Where the line is, as for tags |

These are never tags: the profile's tags stay in `tags`, with their statuses. A text read
without a pattern is less reliable; treat `other_texts` as suggestions for a person.

## How likely a line is a tag

For now `tag_likelihood` comes from generic cues, true of specimen tags in most
collections (`tagsort.pipeline.openread`):

- specimen numbers hold digits, are short (3 to 16 characters) and have few words;
- a few capital letters before the digits (`GJ07966`, `NMC16368`) and a dash or slash
  between digits (`15950-1`, `12/345`) make a line more likely a tag;
- printed words without digits are headers (`NATIONAL MUSEUM OF CANADA`);
- rows of short numbers are rulers or scales, single characters are stray marks.

These cues are a starting point: a selector trained on lines people kept and discarded
will replace them, measured on held-out datasets like every model.

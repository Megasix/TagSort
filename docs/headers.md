# Printed headers

Many collections use pre-printed tags: a header printed on every tag, such as
`NATIONAL MUSEUM OF CANADA` or `HERBARIUM B`, and a number written by hand. The number
alone may fit several kinds of tag in a profile (five digits, for instance); the header
says which kind it is.

## In a profile

Give the kind an optional `header`:

```json
{
  "schema_version": "1.0",
  "name": "Museum A",
  "tags_per_individual": 1,
  "tags": [
    { "id": "museum_a", "pattern": "\\d{5}", "header": "NATIONAL MUSEUM OF CANADA" },
    { "id": "loan", "pattern": "\\d{5}" }
  ]
}
```

`header` is optional: profiles without it read exactly as before.

## How it is used

The local pipeline reads every text line of the photo, not only the ones that look like
tags. A header counts as seen when the photo's text contains it, ignoring case, spaces,
punctuation and line breaks, and allowing a few recognition errors (similarity of at
least 0.8, see `HEADER_SIMILARITY`). A header split over several lines is found too.

When a reading fits several kinds, TagSort reports, in this order:

1. a kind whose header was seen in the photo;
2. a kind without a header;
3. a kind whose header was not seen (so a tag is still read when its header is hidden or
   unreadable).

Within each group, the profile's order decides. A reading that fits only one kind keeps
it. The header itself is never reported as a tag: applications decide what to do with
the kind, for instance put a short prefix such as `NMC` in front of the number.

## Limits

- One header decision per photo: a photo showing tags of two collections with the same
  number shape needs one photo per tag.
- The vision API fallback reads only the cropped tag; it does not use headers.

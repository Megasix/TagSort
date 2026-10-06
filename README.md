# TagSort

Open-source engine that reads handwritten or printed specimen ID tags in photos, in any orientation. Returns tag text, position, angle and a calibrated confidence score. Lightweight ONNX core, versioned JSON contracts and pluggable backends, built to embed in mobile, web and research apps.

[![CI](https://github.com/Megasix/TagSort/actions/workflows/ci.yml/badge.svg)](https://github.com/Megasix/TagSort/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Megasix/TagSort/blob/main/LICENSE)

TagSort runs on your machine with small open models (6 MB by default): no photo leaves the
device unless you add a vision API for the tags it is unsure of. It reports a tag as
`accepted` only when the reading fits the patterns you give it; everything else goes to
`review`, so a wrong number is never accepted silently.

## Read a tag in five minutes

You need Python 3.11 or later.

<!-- quickstart -->
```sh
# 1. Install TagSort and download the default model (6 MB, checked by SHA-256).
pip install "tagsort[api]"
tagsort models download

# 2. Describe your tags: here, "GJ" followed by five digits.
cat > profile.json <<'EOF'
{
  "schema_version": "1.0",
  "name": "My collection",
  "tags_per_individual": 1,
  "tags": [{ "id": "catalog", "pattern": "GJ\\d{5}" }]
}
EOF

# 3. Read a photo (here, a sample image) and print the result as JSON.
curl -sLO https://raw.githubusercontent.com/Megasix/TagSort/main/reference/images/upright.png
tagsort read upright.png --profile profile.json
```

The result lists every tag found, with its text, confidence, status, position and angle:

```json
{
  "file": "upright.png",
  "result": {
    "schema_version": "1.0",
    "engine_version": "0.1.0",
    "model_version": "local:ppocrv6-tiny-1.0",
    "image": {
      "width": 1600,
      "height": 1200,
      "exif_rotation": 0
    },
    "tags": [
      {
        "tag_id": "catalog",
        "text": "GJ07966",
        "confidence": 0.9847,
        "status": "accepted",
        "polygon": [[385.5, 384.7], [739.5, 388.4], [738.2, 508.5], [384.2, 504.7]],
        "angle": 0.6,
        "candidates": [
          {
            "text": "GJ00966",
            "confidence": 0.0002
          },
          {
            "text": "GJ07066",
            "confidence": 0.0001
          }
        ],
        "source": "local"
      }
    ],
    "timings_ms": {
      "preprocess": 16,
      "local": 265
    }
  }
}
```

Run `tagsort read` on a folder to read every photo in it. From Python:

```python
from tagsort import Profile, Reader

reader = Reader(Profile.from_file("profile.json"))  # local model, no network
result = reader.read("upright.png")  # path, bytes, PIL image or array
for tag in result.tags:
    print(tag.tag_id, tag.text, tag.confidence, tag.status)

for result in reader.read_batch(paths, workers=8):  # many photos, in order
    ...
```

## Choosing how to read

| Configuration | Good for | Network |
| --- | --- | --- |
| `Reader(profile)`: local `ppocrv6-tiny` model | Mobile and field use, printed tags | None |
| `Reader(profile, backend=LocalPipeline("ppocrv6-small"))` | Desktops and servers | None |
| Local model with `fallback=GeminiProvider(api_key=...)` | Older or handwritten tags | Only the crop of doubtful tags |
| `Reader(profile, backend=GeminiProvider(api_key=...))` | No local model at all | Whole photos (EXIF and GPS removed) |

On the reference set (160 photos, 60 tags), every configuration made zero silent errors;
`ppocrv6-medium` read 60 of 60 locally, and Gemini Flash-Lite read 60 of 60 for about
0.51 USD per 1,000 photos. Details:
[local models](https://github.com/Megasix/TagSort/blob/main/docs/local.md),
[vision API providers](https://github.com/Megasix/TagSort/blob/main/docs/providers.md),
[model card](https://github.com/Megasix/TagSort/blob/main/docs/model-card.md).

## Profiles

A profile describes the tags of one collection: what each kind of tag may say.

```json
{
  "schema_version": "1.0",
  "name": "Museum A",
  "tags_per_individual": 2,
  "tags": [
    { "id": "primary", "pattern": "MD\\d{5}" },
    { "id": "secondary", "pattern": "[A-Z]{2}-\\d{3,4}" }
  ]
}
```

Patterns use a small regex subset (literals, classes, `\d`, `{n}`, `{n,m}`, alternation)
that compiles to a finite automaton; an invalid profile is rejected when it is loaded,
with the location of the problem. See
[docs/patterns.md](https://github.com/Megasix/TagSort/blob/main/docs/patterns.md).

## Running as a service

`tagsort serve`, or the Docker image, exposes an HTTP API for applications in any language,
described in [`schemas/api.v1.openapi.json`](https://github.com/Megasix/TagSort/blob/main/schemas/api.v1.openapi.json).
See [docs/server.md](https://github.com/Megasix/TagSort/blob/main/docs/server.md) for
configuration and deployment, for example on Render.

```sh
docker run -p 8000:8000 -e TAGSORT_API_TOKEN=change-me ghcr.io/megasix/tagsort
docker run -v "$PWD:/data" ghcr.io/megasix/tagsort read /data/photos --profile /data/profile.json
```

## Measuring accuracy on your photos

`tagsort-eval` reads a labeled folder of your photos and reports exact matches, silent
errors, review load, cost and speed. `tagsort-eval prelabel` lets AI fill in the labels for
a person to check. See [docs/evaluation.md](https://github.com/Megasix/TagSort/blob/main/docs/evaluation.md).

## Contracts and ports

Inputs and outputs follow versioned JSON Schemas, so applications in Dart, TypeScript or any
other language can use them without reading the Python code:
[profiles](https://github.com/Megasix/TagSort/blob/main/schemas/profile.v1.json),
[results](https://github.com/Megasix/TagSort/blob/main/schemas/result.v1.json) and the
[HTTP API](https://github.com/Megasix/TagSort/blob/main/schemas/api.v1.openapi.json).
Reference vectors check ports of the
[pattern grammar](https://github.com/Megasix/TagSort/blob/main/reference/grammar.v1.json)
and of the [local pipeline](https://github.com/Megasix/TagSort/blob/main/reference/pipeline.v1.json).

## Privacy

TagSort never sends telemetry and never uses the network on its own: models are downloaded
only when you ask, and photos or crops go to a vision API only when you configure one, with
your own key.

## Development

TagSort uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync --all-extras
uv run tagsort models download
uv run pytest
```

See [CONTRIBUTING.md](https://github.com/Megasix/TagSort/blob/main/CONTRIBUTING.md) for the
quality checks, commit style and dependency license policy, and
[CHANGELOG.md](https://github.com/Megasix/TagSort/blob/main/CHANGELOG.md) for what changed.

## License

Apache License 2.0. See [LICENSE](https://github.com/Megasix/TagSort/blob/main/LICENSE) and
[NOTICE](https://github.com/Megasix/TagSort/blob/main/NOTICE). The default models are
PP-OCRv6 by PaddlePaddle, also under Apache 2.0.

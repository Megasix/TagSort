# TagSort as a service

Applications that do not run Python call TagSort over HTTP. The contract is
[`schemas/api.v1.openapi.json`](../schemas/api.v1.openapi.json) (OpenAPI 3.1); responses
follow [`result.v1.json`](../schemas/result.v1.json). Clients can generate their code from
that file.

Photos are read in memory and never stored. Logs hold counts and timings only, never
photos, texts or tokens.

## Endpoints

| Method and path | Auth | Purpose |
| --- | --- | --- |
| `GET /v1/health` | none | Status, engine version, model, fallback |
| `GET /v1/models` | bearer | Models the server knows and whether they are downloaded |
| `POST /v1/read` | bearer | Read one photo (`multipart/form-data`: `image`, and `profile` or `profile_name`) |

```sh
curl -H "Authorization: Bearer $TOKEN" \
     -F image=@IMG_0412.jpg \
     -F "profile=<museum_a.json" \
     https://tagsort.example.org/v1/read
```

Errors have a stable shape, `{"error": {"code": "...", "message": "..."}}`, with codes
`invalid_request`, `invalid_profile`, `invalid_pattern` (with `location` and the grammar's
`pattern_code`), `invalid_image`, `unauthorized`, `unknown_profile`, `too_large`,
`model_unavailable` and `internal`.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `TAGSORT_API_TOKEN` | required | Bearer token clients send. Use a long random value. |
| `TAGSORT_ALLOW_NO_TOKEN` | unset | `1` disables the token, for local development only |
| `TAGSORT_MODEL` | `ppocrv6-small` | Local model (the Docker image holds `ppocrv6-tiny` and `ppocrv6-small`) |
| `TAGSORT_FALLBACK` | off | Vision API for doubtful tags, `provider[:model]`, for example `gemini`; only tag crops are sent |
| `GEMINI_API_KEY`, ... | | Key of the fallback provider |
| `TAGSORT_PROFILES` | unset | Folder of profiles (`*.json`), usable as `profile_name` |
| `TAGSORT_MAX_UPLOAD_MB` | 25 | Largest accepted photo |
| `TAGSORT_CONCURRENCY` | number of CPUs | Photos read at the same time |
| `PORT` | 8000 | Listening port (set by most hosts) |

## Running

With Docker, no Python needed:

```sh
docker build -f docker/Dockerfile -t tagsort .
docker run -p 8000:8000 -e TAGSORT_API_TOKEN=change-me tagsort
docker run -v "$PWD:/data" tagsort read /data/photos --profile /data/profile.json
```

Without Docker:

```sh
pip install "tagsort[server,api]"
tagsort models download ppocrv6-small
TAGSORT_API_TOKEN=change-me tagsort serve --host 0.0.0.0
```

`tagsort read` prints one JSON line per photo, `{"file": ..., "result": ...}` or
`{"file": ..., "error": ...}`, and keeps going when a photo fails.

## Resources

Reading needs a CPU only; GPUs are only for training. Measured on a laptop (Apple
silicon), one photo at a time; expect cloud CPUs to be two to three times slower.

| Model | Peak memory | Time per photo |
| --- | --- | --- |
| `ppocrv6-tiny` | about 750 MB | 0.5 s |
| `ppocrv6-small` (server default) | about 1 GB | 0.9 s |
| `ppocrv6-medium` | about 1.3 GB | 2.9 s |

Plan about 1 GB per photo read at the same time on top of the base, and set
`TAGSORT_CONCURRENCY` to fit the instance's memory.

## Deploying on Render

1. In Render, choose **New > Web Service** and connect the GitHub repository.
2. **Language**: Docker. **Dockerfile path**: `docker/Dockerfile`. Render builds the image,
   including the model download.
3. **Instance type**: at least 2 GB of memory for `ppocrv6-small` with
   `TAGSORT_CONCURRENCY=1`; 4 GB and 2 CPUs for two photos at a time. Check Render's
   current prices.
4. **Environment**: `TAGSORT_API_TOKEN` (a long random value, also stored in the calling
   application), and optionally `TAGSORT_FALLBACK=gemini` with `GEMINI_API_KEY`.
5. **Health check path**: `/v1/health`.

Render serves HTTPS and sets `PORT`. Keep the token secret: anyone holding it can send
photos to the service. Applications such as a web front end should call TagSort from
their own server, never from the browser with the token.

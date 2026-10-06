# TagSort HTTP server and CLI, with the tiny and small models baked in.
#
#   docker build -f docker/Dockerfile -t tagsort .
#   docker run -p 8000:8000 -e TAGSORT_API_TOKEN=... tagsort            # HTTP server
#   docker run -v "$PWD:/data" tagsort read /data/photos --profile /data/profile.json
#
# Configuration: see docs/server.md. No photo, key or token is ever stored in the image.

FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir "uv==0.12.*"
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra server --extra api
# Models are downloaded here, explicitly, and checked against their manifest SHA-256.
ENV TAGSORT_MODELS=/models
RUN /app/.venv/bin/tagsort models download ppocrv6-tiny \
 && /app/.venv/bin/tagsort models download ppocrv6-small

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/Megasix/TagSort" \
      org.opencontainers.image.description="Reads specimen ID tags in photos" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN useradd --create-home --uid 10001 tagsort
COPY --from=build /app/.venv /app/.venv
COPY --from=build /models /models
ENV PATH="/app/.venv/bin:$PATH" \
    TAGSORT_MODELS=/models \
    TAGSORT_MODEL=ppocrv6-small \
    PORT=8000 \
    PYTHONUNBUFFERED=1
USER tagsort
WORKDIR /home/tagsort
EXPOSE 8000
ENTRYPOINT ["tagsort"]
CMD ["serve", "--host", "0.0.0.0"]

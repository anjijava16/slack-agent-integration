# ---------- build stage: compile wheels ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---------- runtime stage ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    # Default for `docker run` on a laptop. Cloud Run overrides APP_ENV, DATABASE_URL
    # (Cloud SQL) and the Qdrant settings (Qdrant Cloud).
    APP_ENV=local \
    DATABASE_URL=sqlite:////tmp/data/agent.db \
    # Models live in the image, so Cloud Run cold starts never download anything.
    EMBED_CACHE_DIR=/opt/models \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    HF_HUB_DISABLE_TELEMETRY=1

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels

# Bake the default models in (set PRELOAD_MODELS=0 to skip, e.g. when using EMBED_PROVIDER=openai):
#   dense embedding BAAI/bge-small-en-v1.5, the local cross-encoder, and the tokenizer the text splitter uses.
ARG PRELOAD_MODELS=1
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
RUN if [ "$PRELOAD_MODELS" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBED_MODEL}', cache_dir='/opt/models')" && \
      python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2', cache_dir='/opt/models')" && \
      python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" ; \
    fi && mkdir -p /opt/models /opt/tiktoken && chmod -R a+rX /opt/models /opt/tiktoken

COPY app ./app
COPY config ./config

USER appuser
EXPOSE 8080
# One worker per container: Cloud Run scales by containers, and the async app
# handles many concurrent Slack events in one process. Shell form so $PORT expands.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips="*" --timeout-keep-alive 75

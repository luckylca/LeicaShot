FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LAIKA_API_HOST=0.0.0.0 \
    LAIKA_API_PORT=8000 \
    LAIKA_API_STORAGE=/var/lib/laika/tasks

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --no-compile '.[api,inference]' \
    && groupadd --system --gid 10001 laika \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent \
        --shell /usr/sbin/nologin laika \
    && mkdir -p /var/lib/laika/tasks \
    && chown -R 10001:10001 /var/lib/laika

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3).read()"

ENTRYPOINT ["laika-api"]

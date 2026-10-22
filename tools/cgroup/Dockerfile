FROM python:3.11-slim-bookworm

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --disable-pip-version-check --no-cache-dir ".[inference]"

COPY scripts/benchmark_processing_pipeline.py ./scripts/

ENTRYPOINT ["python", "scripts/benchmark_processing_pipeline.py"]

# Build from repository root:
#   docker build -f deploy/Dockerfile -t mft-cashcow-paper .
#
# Expects v2/requirements.txt (same stack as local venv).

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# TA-Lib C library + toolchain for any wheels that compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libta-lib0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY v2/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["python", "v3/scripts/run_paper_loop.py", "--iterations", "0"]

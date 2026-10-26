# CPU-only app server. All models are remote: Meta Model API for reasoning,
# deploy/modal_sam.py for SAM. Can run on the Mac or any small cloud VM.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m forgemind
WORKDIR /app
COPY pyproject.toml README.md ./
COPY forgemind ./forgemind
RUN pip install --no-cache-dir ".[vision]"
USER forgemind
ENV FM_DATA_DIR=/data FM_HOST=0.0.0.0 FM_PORT=8770
VOLUME /data
EXPOSE 8770
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8770/healthz')"
CMD ["forgemind", "serve"]

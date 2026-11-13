FROM python:3.12-slim AS coordinator
ENV PYTHONUNBUFFERED=1 ROLE=coordinator
WORKDIR /app
RUN useradd --uid 10001 --create-home app && mkdir -p /data/output && chown -R 10001:10001 /data/output
COPY server.py model_files.py index.html ./
USER 10001:10001
CMD ["python3", "server.py"]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS worker
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/data/cache
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 && pip install --no-cache-dir --no-compile -r requirements.txt
COPY server.py model_files.py skill_t2i.py index.html lora_library.json ./
RUN useradd --uid 10001 --create-home app && mkdir -p /data/models /data/output /data/cache && chown -R 10001:10001 /data
ENV ROLE=worker PORT=8188
USER 10001:10001
CMD ["python3", "server.py"]

FROM nvidia/cuda:13.3.0-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    ELY_EYE_HOME=/data/ely-eye \
    ELY_EYE_RUNTIME_BACKEND=sglang \
    ELY_EYE_SGLANG_BASE_URL=http://127.0.0.1:8000/v1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip nodejs npm ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend ./backend
COPY frontend ./frontend
COPY README.md PRD.md ./

RUN python3.12 -m venv /opt/ely-eye \
    && /opt/ely-eye/bin/pip install --upgrade pip setuptools wheel \
    && /opt/ely-eye/bin/pip install -e ./backend \
    && cd frontend \
    && npm install \
    && npm run build

EXPOSE 8765
CMD ["/opt/ely-eye/bin/ely-eye", "serve", "--host", "0.0.0.0", "--port", "8765"]

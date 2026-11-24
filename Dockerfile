# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.5.0-cudnn-devel-ubuntu26.04

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=hardlink \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git \
        libibverbs1 libnuma1 librdmacm1 libsndfile1 libsox-fmt-all \
        ninja-build pkg-config sox \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/
RUN uv python install 3.12 \
    && uv venv --python 3.12 "${VIRTUAL_ENV}" \
    && uv cache clean

WORKDIR /workspace/sglang-omni
COPY . ./
RUN uv pip install --prerelease=allow -v -e . oss2==2.19.1 python-dotenv==1.2.1 \
    && uv cache clean \
    && rm -rf /root/.cache/pip
COPY scripts/download_model.py /opt/scripts/download_model.py

# Supply the download step and the inference command at container startup.

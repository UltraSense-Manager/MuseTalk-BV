# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.2
ARG PASSWORD
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/app/hf
ENV XDG_CACHE_HOME=/app/hf_cache
ENV POOCH_CACHE=/app/p_cache
ENV NUMBA_CACHE_DIR=/tmp
WORKDIR /app

# Install base dependencies
RUN apt-get update && apt-get install -y -q --no-install-recommends \
        apt-transport-https \
        build-essential \
        ca-certificates \
        curl \
        git \
        libssl-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

RUN apt update && apt install -y gcc clang clang-tools cmake python3
# Update package lists and install necessary dependencies for Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-dev && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Create a symbolic link for 'python' to point to 'python3.10'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Verify Python and pip installations
RUN python3 --version
RUN pip --version

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
# default cpu: https://download.pytorch.org/whl/cpu 
# rocm utilizes AMD GPU acceleration to provide similar performance to CUDA

ARG UID=10001
RUN apt-get update && apt-get install -y git

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install --upgrade setuptools
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser


# Copy the source code into the container.
COPY . .

RUN rm -rf /app/src

RUN chown -R appuser /app

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt 

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

RUN rm -rf /venv/MuseTalk/lib/python3.10/site-packages/mmcv*

RUN pip uninstall -y mmcv mmcv-full mmcv-lite 
RUN pip install --upgrade pip setuptools wheel
RUN pip install "setuptools==60.2.0"
RUN pip install chumpy==0.70 --no-build-isolation
RUN pip install mmpose openmim --no-build-isolation
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine --no-build-isolation
RUN pip install mmcv==2.0.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
  --no-cache-dir --no-build-isolation
RUN mim install "mmdet==3.1.0" --no-build-isolation
RUN mim install "mmpose==1.1.0" --no-build-isolation

RUN which ffmpeg
RUN sh download_weights.sh
# Switch to the non-privileged user to run the application.
USER appuser

RUN ls 
#Test run
RUN sh inference.sh v1.5 normal

# Expose the port that the application listens on.
EXPOSE 7860

# Run the application.
CMD python app.py --ip 0.0.0.0 --port 7860

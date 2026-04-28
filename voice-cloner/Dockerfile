# Voice Cloner API - staging deployment
#
# Build:  docker build -t voice-cloner:staging .
# Run:   docker run -p 8000:8000 -e JWT_SECRET=... -e DDB_TABLE_NAME=... -e AWS_REGION=... voice-cloner:staging
#
# Requires checkpoints_v2/converter/config.json and checkpoint.pth in build context
# (or mount a volume over /app/checkpoints_v2 at runtime).

FROM python:3.11-slim

# Avoid torchaudio FFmpeg load error when system FFmpeg is missing or wrong version
ENV TORIO_USE_FFMPEG=0
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for audio (optional; remove if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# App and OpenVoice deps
COPY constraint.txt requirements.txt ./
RUN pip install --no-cache-dir -c constraint.txt "torch" "torchaudio" \
    && pip install --no-cache-dir -c constraint.txt -r requirements.txt \
    && pip install --no-cache-dir "fastapi" "uvicorn[standard]" "boto3" "PyJWT" "python-dotenv" "pydantic"

# OpenVoice package (local)
COPY setup.py ./
COPY openvoice/ ./openvoice/
RUN pip install --no-cache-dir -e .

# Checkpoints (must include checkpoints_v2/converter/config.json and checkpoint.pth)
COPY checkpoints_v2/ ./checkpoints_v2/

# Application code
COPY main.py auth.py audio.py cloner.py log_config.py models.py ./

# Create output dir used by cloner
RUN mkdir -p outputs_v2

EXPOSE 8000

# Override PORT via env if needed
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# SPDX-License-Identifier: MIT
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install voxlocal with server and TTS support
COPY pyproject.toml README.md NOTICE.md ./
COPY voxlocal/ ./voxlocal/
RUN pip install --no-cache-dir ".[server,tts]"

# ─── Runtime image ────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN groupadd --system --gid 1001 voxlocal && \
    useradd --system --uid 1001 --gid voxlocal --create-home --shell /bin/bash voxlocal

# Install runtime dependencies only
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY pyproject.toml NOTICE.md ./

WORKDIR /app

# Pre-download models into the image layer
# This makes the container start faster and avoids download at runtime
RUN python -c "
from voxlocal import VoxLocal
# Pre-download common models
v = VoxLocal(language='en')
v.setup(stt=True, tts=True, warmup_tts=True)
v = VoxLocal(language='es')
v.setup(stt=True, tts=True, warmup_tts=True)
print('Models pre-downloaded successfully')
"

USER voxlocal:voxlocal

EXPOSE 8765

ENV VOXLOCAL_HOST=0.0.0.0 \
    VOXLOCAL_PORT=8765 \
    VOXLOCAL_LOG_LEVEL=INFO

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "
import urllib.request
response = urllib.request.urlopen('http://localhost:8765/v1/health')
assert response.status == 200
body = response.read().decode()
assert '\"status\":\"ok\"' in body.replace(' ', '')
"

ENTRYPOINT ["voxlocal-server"]
CMD ["--host", "0.0.0.0", "--port", "8765"]

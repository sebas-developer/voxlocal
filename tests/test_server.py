import json

import numpy as np
import pytest

from voxlocal._audio import AudioChunk
from voxlocal.server import chunk_to_ndjson, create_app


def test_chunk_ndjson_contract():
    encoded = chunk_to_ndjson(
        AudioChunk(
            sequence=0,
            numpy=np.ones(10, dtype=np.float32),
            sample_rate=24000,
            final=True,
        )
    )

    record = json.loads(encoded)

    assert encoded.endswith(b"\n")
    assert record["type"] == "audio_chunk"
    assert record["encoding"] == "pcm_s16le"
    assert record["final"] is True


def test_server_exposes_versioned_routes():
    pytest.importorskip("fastapi")

    app = create_app()
    paths = {route.path for route in app.routes}

    assert {"/v1/health", "/v1/stt", "/v1/tts", "/v1/tts/stream"} <= paths

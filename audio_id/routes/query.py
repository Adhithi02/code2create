"""POST /query endpoint."""

import logging
import time

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from audio_id.metrics import MATCH_TYPE, QUERIES_TOTAL, QUERY_LATENCY

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryResponse(BaseModel):
    """Query result response model."""
    song_id: str | None
    genre: str | None
    confidence: float
    match_type: str


@router.post("/query")
async def query(request: Request, audio: UploadFile | None = File(default=None)):
    """Identify a song from an audio clip."""
    index = request.app.state.index
    if not index.ready:
        return JSONResponse(status_code=503, content={"detail": "index not ready"})

    QUERIES_TOTAL.inc()
    start = time.monotonic()

    try:
        # Determine how to read the audio bytes
        content_type = request.headers.get("content-type", "")

        if audio is not None:
            # multipart/form-data upload
            audio_bytes = await audio.read()
        elif "multipart/form-data" in content_type:
            # Try to get file from form
            form = await request.form()
            upload = form.get("audio")
            if upload is None:
                raise HTTPException(status_code=400, detail="no audio field in form")
            audio_bytes = await upload.read()
        else:
            # application/octet-stream or raw bytes
            audio_bytes = await request.body()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="empty audio data")

        pipeline = request.app.state.pipeline
        result = pipeline.process(audio_bytes)

        elapsed = time.monotonic() - start
        QUERY_LATENCY.observe(elapsed)
        MATCH_TYPE.labels(type=result["match_type"]).inc()

        return QueryResponse(
            song_id=result["song_id"],
            genre=result["genre"],
            confidence=result["confidence"],
            match_type=result["match_type"],
        )

    except HTTPException:
        elapsed = time.monotonic() - start
        QUERY_LATENCY.observe(elapsed)
        raise
    except Exception as exc:
        elapsed = time.monotonic() - start
        QUERY_LATENCY.observe(elapsed)
        logger.error("Query processing error: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=f"query processing failed: {exc}")

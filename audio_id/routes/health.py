"""GET /health endpoint."""

from pydantic import BaseModel
from fastapi import APIRouter, Request

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    songs_indexed: int
    hash_postings: int
    embeddings_indexed: int
    last_error: str | None


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return system health status."""
    index = request.app.state.index
    status = "ready" if index.ready else "not_ready"
    return HealthResponse(
        status=status,
        songs_indexed=index.songs_count,
        hash_postings=index.postings_count,
        embeddings_indexed=index.embeddings_count,
        last_error=index.last_error,
    )

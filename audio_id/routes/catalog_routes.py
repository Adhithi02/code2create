"""GET /songs endpoint."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class SongResponse(BaseModel):
    """Song catalog entry response model."""
    song_id: str
    genre: str
    path: str


@router.get("/songs", response_model=list[SongResponse])
async def list_songs(request: Request) -> list[SongResponse]:
    """Return all indexed songs."""
    catalog = request.app.state.catalog
    songs = []
    for _, row in catalog.iterrows():
        if row["status"] == "ok":
            songs.append(SongResponse(
                song_id=row["song_id"],
                genre=row["genre"],
                path=row["path"],
            ))
    return songs

"""POST /index/rebuild endpoint."""

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _rebuild_index(request: Request) -> None:
    """Background task to rebuild the index from the store."""
    try:
        index = request.app.state.index
        store = request.app.state.store
        await asyncio.get_event_loop().run_in_executor(None, index.build, store)
        logger.info("Index rebuild completed")
    except Exception as exc:
        logger.error("Index rebuild failed: %s", exc, exc_info=True)


@router.post("/index/rebuild")
async def rebuild_index(request: Request):
    """Trigger a background index rebuild."""
    asyncio.create_task(_rebuild_index(request))
    return JSONResponse(status_code=202, content={"status": "rebuilding"})

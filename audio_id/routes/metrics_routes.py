"""GET /metrics endpoint."""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest

router = APIRouter()


@router.get("/metrics")
async def metrics():
    """Return Prometheus metrics in text format."""
    body = generate_latest()
    return Response(content=body, media_type="text/plain")

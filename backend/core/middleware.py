"""
CEREP Middleware — request timing, structured access logging, error handling.
"""
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.core.logging import get_logger

logger = get_logger("middleware")


class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID and X-Process-Time headers, logs every request."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.error(
                "Request failed",
                extra={"extra": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration * 1000, 1),
                    "error": str(exc),
                }}
            )
            raise

        duration = time.perf_counter() - start
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration:.3f}s"

        logger.info(
            "Request completed",
            extra={"extra": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 1),
            }}
        )
        return response

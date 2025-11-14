"""
FastAPI Middleware Components

Provides middleware for:
- Request logging and monitoring
- Metrics collection (Prometheus)
- CORS configuration
- Error handling
- Request ID tracking
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

# Request counter by endpoint and status code
request_counter = Counter(
    "api_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status_code"],
)

# Request latency histogram
request_latency = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Recommendation request counter
recommendation_counter = Counter(
    "recommendations_served_total",
    "Total number of recommendations served",
    ["endpoint", "cached"],
)

# Cache hit/miss counter
cache_counter = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "result"],
)

# Error counter
error_counter = Counter(
    "api_errors_total",
    "Total number of errors",
    ["error_type", "endpoint"],
)


# ==================== Request ID Middleware ====================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in request state for access in route handlers
        request.state.request_id = request_id

        # Call next middleware/route
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


# ==================== Logging Middleware ====================


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing information."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()

        # Get request ID
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - Starting"
        )

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"[{request_id}] Request failed: {e}", exc_info=True)
            status_code = 500
            raise

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Status: {status_code} - Duration: {duration:.3f}s"
        )

        return response


# ==================== Metrics Middleware ====================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract path template (not individual IDs)
        path = request.url.path
        method = request.method

        # Simplify path for metrics (group by route template)
        endpoint = self._get_endpoint_template(path)

        # Start timer
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            # Increment error counter
            error_type = type(e).__name__
            error_counter.labels(error_type=error_type, endpoint=endpoint).inc()
            raise
        finally:
            # Record latency
            duration = time.time() - start_time
            request_latency.labels(method=method, endpoint=endpoint).observe(duration)

            # Record request count
            request_counter.labels(
                method=method, endpoint=endpoint, status_code=status_code
            ).inc()

        return response

    @staticmethod
    def _get_endpoint_template(path: str) -> str:
        """
        Convert path to endpoint template for metrics grouping.

        Example: /api/v1/recommend/123 -> /api/v1/recommend/{user_id}
        """
        parts = path.split("/")

        # Replace numeric IDs with template variables
        template_parts = []
        for part in parts:
            if part.isdigit():
                template_parts.append("{id}")
            else:
                template_parts.append(part)

        return "/".join(template_parts)


# ==================== Error Handling Middleware ====================


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handler for consistent error responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Get request ID for tracing
            request_id = getattr(request.state, "request_id", "unknown")

            # Log error
            logger.error(
                f"[{request_id}] Unhandled exception: {type(e).__name__}: {e}",
                exc_info=True,
            )

            # Return JSON error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "detail": str(e) if logger.level <= logging.DEBUG else None,
                    "error_code": "INTERNAL_ERROR",
                    "request_id": request_id,
                },
            )


# ==================== CORS Middleware Configuration ====================


def get_cors_middleware():
    """
    Get configured CORS middleware for frontend integration.

    Returns:
        CORSMiddleware instance
    """
    # TODO: In production, restrict to specific origins
    return CORSMiddleware(
        allow_origins=[
            "http://localhost:3000",  # Next.js dev server
            "http://localhost:8000",  # Local API
            "https://*.vercel.app",  # Vercel deployments
        ],
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
        expose_headers=["X-Request-ID"],  # Expose custom headers to frontend
    )


# ==================== Rate Limiting (Optional) ====================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    NOTE: For production, use Redis-based rate limiting for distributed systems.
    This is a basic implementation for demonstration.
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # IP -> list of timestamps

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health/deep"]:
            return await call_next(request)

        # Check rate limit
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                ts for ts in self.requests[client_ip] if ts > minute_ago
            ]
        else:
            self.requests[client_ip] = []

        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                },
            )

        # Record request
        self.requests[client_ip].append(now)

        # Process request
        return await call_next(request)


# ==================== Middleware Setup Helper ====================


def setup_middleware(app):
    """
    Configure all middleware for the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Order matters! Middleware is executed in reverse order of addition

    # 1. Error handling (outermost - catches all errors)
    app.add_middleware(ErrorHandlerMiddleware)

    # 2. CORS (before request processing)
    cors_config = get_cors_middleware()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allow_methods,
        allow_headers=cors_config.allow_headers,
        expose_headers=cors_config.expose_headers,
    )

    # 3. Rate limiting (optional - comment out if not needed)
    # app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

    # 4. Metrics collection
    app.add_middleware(MetricsMiddleware)

    # 5. Logging
    app.add_middleware(LoggingMiddleware)

    # 6. Request ID (innermost - first to process)
    app.add_middleware(RequestIDMiddleware)

    logger.info("Middleware configured successfully")

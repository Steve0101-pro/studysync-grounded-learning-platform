"""Rate limiting configuration for API protection."""

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiting(app: FastAPI) -> None:
    """
    Configure rate limiting for the FastAPI application.
    
    Rate limit policies:
    - Authentication endpoints: 5 requests per minute
    - Document upload: 10 requests per minute
    - Multimodal: 5 requests per minute 
    - General API: 100 requests per minute
    
    """
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request, exc):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
        )


# Define rate limit keys for different endpoints
RATE_LIMIT_AUTH = "5/minute"
RATE_LIMIT_DOCUMENTS = "10/minute"
RATE_LIMIT_MULTIMODAL = "5/minute"
RATE_LIMIT_STUDY = "10/minute"
RATE_LIMIT_GENERAL = "100/minute"

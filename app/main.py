import logging
import asyncio
import anyio
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.logging_config import setup_logging
from app.middleware.observability import RequestMetricsMiddleware
from app.api.documents import router as documents_router
from app.api.multimodal_router import router as multimodal_router
from app.api.auth import router as auth_router
from app.api.monitoring import router as monitoring_router
from app.api.feedback import router as feedback_router
from app.api.learning import (
    assessments_router,
    courses_router,
    dashboard_router,
    sources_router,
    tutor_router,
)
from app.rate_limit import setup_rate_limiting
from app.api.redis_service import close_redis, check_redis_health, CacheMetrics
from app.services.portkeys_service import portkeys_client
from app.services.model_runtime import close_provider_clients
from app.config import settings
from app.database import check_database_connection
from app.services.neo4j_service import close_neo4j, neo4j_health

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)
STARTUP_DEPENDENCY_TIMEOUT_SECONDS = 5.0


async def _check_startup_dependency(name: str, check) -> bool:
    try:
        return await asyncio.wait_for(
            check(),
            timeout=STARTUP_DEPENDENCY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "%s startup health check timed out after %.1f seconds",
            name,
            STARTUP_DEPENDENCY_TIMEOUT_SECONDS,
        )
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Schema changes are applied separately with `alembic upgrade head`.
    logger.info("Starting application...")
    
    # Test Redis connection
    redis_healthy, portkeys_healthy = await asyncio.gather(
        _check_startup_dependency("Redis", check_redis_health),
        _check_startup_dependency("Portkeys", portkeys_client.health_check),
    )
    if redis_healthy:
        logger.info("Redis connection established and healthy")
    else:
        logger.warning("Redis is not available; caching will be disabled gracefully")
    
    if portkeys_healthy:
        logger.info("Portkeys connection established and healthy")
    elif portkeys_client.enabled:
        logger.warning("Portkeys is configured but not available; LLM monitoring will be unavailable")
    else:
        logger.info("Portkeys monitoring is disabled")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await close_redis()
    close_neo4j()
    await close_provider_clients()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Document Intelligence API",
    version="1.0.0",
    description="API for intelligent document processing with multimodal AI capabilities",
    lifespan=lifespan,
)

setup_rate_limiting(app)
app.add_middleware(RequestMetricsMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check called")
    return {
        "message": "Document Intelligence API is running",
        "status": "healthy"
    }


@app.get("/health/redis", tags=["Health"])
async def redis_health_check():
    """Redis connectivity and cache statistics."""
    redis_is_healthy = await _check_startup_dependency(
        "Redis",
        check_redis_health,
    )
    cache_stats = CacheMetrics.get_stats()
    return {
        "redis_status": "healthy" if redis_is_healthy else "unhealthy",
        "cache_metrics": cache_stats,
    }


@app.get("/health/full", tags=["Health"])
async def full_health_check():
    """Comprehensive health check including all services."""
    async def check_graph() -> dict[str, str | bool]:
        try:
            return await asyncio.wait_for(
                anyio.to_thread.run_sync(neo4j_health, abandon_on_cancel=True),
                timeout=STARTUP_DEPENDENCY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Neo4j health check timed out after %.1f seconds",
                STARTUP_DEPENDENCY_TIMEOUT_SECONDS,
            )
            return {
                "configured": bool(settings.NEO4J_URI.strip()),
                "status": "unhealthy",
            }

    database_is_healthy, redis_is_healthy, graph_status = await asyncio.gather(
        _check_startup_dependency("Database", check_database_connection),
        _check_startup_dependency("Redis", check_redis_health),
        check_graph(),
    )
    cache_stats = CacheMetrics.get_stats()
    return {
        "api_status": "healthy" if database_is_healthy else "degraded",
        "database_status": "healthy" if database_is_healthy else "unhealthy",
        "neo4j": graph_status,
        "redis_status": "healthy" if redis_is_healthy else "unhealthy",
        "cache_metrics": cache_stats,
    } if database_is_healthy else JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "api_status": "degraded",
            "database_status": "unhealthy",
            "neo4j": graph_status,
            "redis_status": "healthy" if redis_is_healthy else "unhealthy",
            "cache_metrics": cache_stats,
        },
    )


# Include routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(multimodal_router)
app.include_router(monitoring_router)
app.include_router(feedback_router)
app.include_router(courses_router)
app.include_router(sources_router)
app.include_router(tutor_router)
app.include_router(assessments_router)
app.include_router(dashboard_router)

logger.info("Application routers configured")
logger.info("Available endpoints: /, /health/redis, /health/full, /monitoring/*, /docs (Swagger UI)")

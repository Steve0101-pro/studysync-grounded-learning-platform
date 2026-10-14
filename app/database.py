import logging

from sqlalchemy.engine import URL, make_url
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)


def normalize_database_url(database_url: str) -> URL:
    """Normalize PostgreSQL URLs for asyncpg and require TLS for Supabase."""
    url = make_url(database_url)
    if url.drivername in {"postgres", "postgresql", "postgresql+psycopg2"}:
        url = url.set(drivername="postgresql+asyncpg")
    if url.drivername != "postgresql+asyncpg":
        raise ValueError("DATABASE_URL must use PostgreSQL with the asyncpg driver")

    if url.host and (
        url.host.endswith(".pooler.supabase.com")
        or url.host.endswith(".supabase.co")
    ):
        query = dict(url.query)
        ssl_mode = query.pop("sslmode", None)
        if "ssl" not in query:
            query["ssl"] = ssl_mode or "require"
        url = url.set(query=query)
    return url


DATABASE_URL = normalize_database_url(settings.DATABASE_URL.get_secret_value())

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
        "timeout": settings.DATABASE_CONNECT_TIMEOUT_SECONDS,
    },
)

SessionLocal = sessionmaker(
     bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    async with SessionLocal() as db:
        try:
            yield db
        except Exception:
            await db.rollback()
            raise


async def check_database_connection() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        return False
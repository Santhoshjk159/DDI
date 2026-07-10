from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from app.config import get_settings

settings = get_settings()

import ssl
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

def create_db_engine(database_url: str, echo: bool = False):
    """
    Creates a SQLAlchemy async engine, cleaning up any unsupported query parameters
    for asyncpg (like sslmode and channel_binding) and configuring SSL correctly.
    """
    if "postgresql+asyncpg" in database_url:
        parsed = urlparse(database_url)
        query_params = dict(parse_qsl(parsed.query))
        
        connect_args = {}
        
        # Remove unsupported parameters for asyncpg connection
        sslmode = query_params.pop("sslmode", None)
        query_params.pop("channel_binding", None)
        
        # If sslmode is specified or this is a Neon database, enable SSL
        if sslmode in ("require", "verify-ca", "verify-full") or "neon.tech" in (parsed.hostname or ""):
            ssl_context = ssl.create_default_context()
            if sslmode == "require" or not sslmode:
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            connect_args["ssl"] = ssl_context
            
        new_query = urlencode(query_params)
        cleaned_url = urlunparse(parsed._replace(query=new_query))
        
        return create_async_engine(
            cleaned_url,
            echo=echo,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
            connect_args=connect_args
        )
        
    return create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )

engine = create_db_engine(
    settings.database_url,
    echo=settings.app_env == "development",
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    from app.models import drug, prediction  # noqa: F401
    async with engine.begin() as conn:
        # Enable trigram extension for fuzzy search (safe if already exists)
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        except Exception:
            pass  # Extension may not be available on all PostgreSQL setups
        await conn.run_sync(Base.metadata.create_all)

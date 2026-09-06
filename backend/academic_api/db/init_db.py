import asyncio

from backend.academic_api.db.base import Base
from backend.academic_api.db.session import engine
from backend.academic_api.models import EvaluationJob, Paper


async def create_tables() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("Academic database tables created.")


if __name__ == "__main__":
    asyncio.run(create_tables())
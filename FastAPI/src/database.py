from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from collections.abc import Generator
from fastapi_users.db import SQLAlchemyUserDatabase
from src.models import User, Base

DATABASE_URL = "sqlite:///./blog.db"

#sync engine (CLI, alembic) -> When greenlet loading fails(AVD)
sync_engine = create_engine(
    DATABASE_URL, 
    echo=True,
    connect_args = {"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = sync_engine
)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=sync_engine)
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from sqlalchemy import Engine

from sqlmodel import SQLModel, Session, create_engine

ENGINE: Engine | None = None


def init_engine(root: Path):
    global ENGINE
    data = root / "db"
    data.mkdir(parents=True, exist_ok=True)
    ENGINE = create_engine(
        f"sqlite:///{data}/bip1.db", connect_args={"check_same_thread": False}
    )


def create_db_and_tables():
    if ENGINE is None:
        raise RuntimeError("Engine not initialized. Call init_engine first.")
    SQLModel.metadata.create_all(ENGINE)


def get_session():
    if ENGINE is None:
        raise RuntimeError("Engine not initialized. Call init_engine first.")
    with Session(ENGINE) as s:
        yield s


SessionDep = Annotated[Session, Depends(get_session)]

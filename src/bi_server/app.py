from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from bi_server.config import ROOT_PATH
from bi_server.db.state import SessionDep, create_db_and_tables, init_engine
from bi_server.dto.train import TrainData, TrainResult
from bi_server.ml.state import nltk_required_download
from bi_server.service.ml_model import MLModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    nltk_required_download(ROOT_PATH)

    init_engine(ROOT_PATH)
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/infer")
def inference(texts: list[str], db: SessionDep) -> list[int]:
    service = MLModelService(db)
    return service.infer_text(texts)


@app.post("/retrain")
def retrain(data: list[TrainData], db: SessionDep) -> TrainResult:
    service = MLModelService(db)
    return service.retrain_model(data)

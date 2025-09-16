import logging

from sqlmodel import Session, select

from bi_server.dto.train import TrainData, TrainResult
from bi_server.ml.service import infer, retrain
from bi_server.db.models import MLModel

logger = logging.getLogger(__name__)


class MLModelService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def _get_latest_model(self) -> int | None:
        try:
            logger.debug("Querying for latest model version")
            statement = select(MLModel).order_by(MLModel.version.desc()).limit(1)
            latest_model = self.db.exec(statement).first()

            if latest_model:
                logger.info(f"Latest model version found: {latest_model.version}")
                return latest_model.version
            else:
                logger.info("No models found in database")
                return None

        except Exception as e:
            logger.error(f"Unexpected error while querying latest model: {e}")
            raise RuntimeError(f"Unexpected database error: {e}") from e

    def retrain_model(self, data: list[TrainData]) -> TrainResult:
        latest_version = self._get_latest_model()
        new_version = latest_version + 1 if latest_version else 1

        result = retrain(data, new_version)

        model = MLModel(
            version=new_version,
            precision=result.precision,
            recall=result.recall,
            f1score=result.f1score,
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        logger.info(f"Model version {new_version} saved to database successfully")
        return result

    def infer_text(self, texts: list[str]) -> list[int]:
        latest_version = self._get_latest_model()
        if latest_version is None:
            raise RuntimeError(
                "No trained model available. Please retrain a model first."
            )

        predictions = infer(texts, latest_version)

        return predictions

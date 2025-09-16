import logging

import joblib

import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from bi_server.config import ROOT_PATH
from bi_server.dto.train import TrainData, TrainResult
from bi_server.ml.model import create_model

logger = logging.getLogger(__name__)


def infer(data: list[str], v: int) -> list[int]:
    model_path = ROOT_PATH / "models" / f"modelv{v}"
    logger.info(f"Loading model from {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    if not isinstance(model, Pipeline):
        error_msg = f"Expected Pipeline model, got {type(model)}"
        logger.error(error_msg)
        raise TypeError(error_msg)

    logger.debug(f"Making predictions for {len(data)} samples")
    try:
        predictions = model.predict(data)
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise RuntimeError(f"Model prediction failed: {e}")

    if not isinstance(predictions, np.ndarray):
        raise TypeError(f"Expected ndarray predictions, got {type(predictions)}")

    result = predictions.tolist()
    logger.info(f"Successfully generated {len(result)} predictions")
    return result


def prepare_training_data(data: list[TrainData]) -> tuple[np.ndarray, np.ndarray]:
    texts = np.array([item.text for item in data], dtype=np.dtypes.StringDType())
    labels = np.array([item.label for item in data], dtype=np.uint8)

    logger.debug(f"Prepared {len(texts)} texts and {len(labels)} labels")
    return texts, labels


def retrain(data: list[TrainData], v: int) -> TrainResult:
    if v < 0:
        raise ValueError("Model version must be non-negative")
    if len(data) < 15:
        raise ValueError("Need at least 15 samples for training with validation split")

    logger.info(f"Starting retraining with {len(data)} samples for model v{v}")

    texts, labels = prepare_training_data(data)

    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    logger.debug(f"Split data: {len(x_train)} train, {len(x_val)} validation")

    training_model = create_model()

    try:
        logger.info("Training model...")
        training_model.fit(x_train, y_train)

        logger.info("Generating predictions on validation set...")
        pred = training_model.predict(x_val)
    except Exception as e:
        logger.error(f"Training or prediction failed: {e}")
        raise RuntimeError(f"Training or prediction failed: {e}")

    precision = precision_score(y_val, pred, average="macro", zero_division=0)
    recall = recall_score(y_val, pred, average="macro", zero_division=0)
    f1 = f1_score(y_val, pred, average="macro", zero_division=0)

    logger.info(
        f"Model metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )

    model_path = ROOT_PATH / "models" / f"modelv{v}"
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(training_model, model_path)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {e}")
        raise RuntimeError(f"Failed to save model to {model_path}: {e}")

    return TrainResult(
        precision=float(precision), recall=float(recall), f1score=float(f1)
    )

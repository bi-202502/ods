from pydantic import BaseModel


class TrainData(BaseModel):
    text: str
    label: int


class TrainResult(BaseModel):
    precision: float
    recall: float
    f1score: float

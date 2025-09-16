from sqlmodel import Field, SQLModel


class MLModel(SQLModel, table=True):
    version: int | None = Field(default=None, primary_key=True)
    precision: float
    recall: float
    f1score: float

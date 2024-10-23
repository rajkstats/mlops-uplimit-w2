from pydantic import BaseModel, ConfigDict, model_validator
from src.constants import LABEL_CLASS_TO_NAME, SentimentLabel

class SimpleModelRequest(BaseModel):
    review: str

class SimpleModelResults(BaseModel):
    NEGATIVE: float
    NEUTRAL: float
    POSITIVE: float
    model_version: str

    @model_validator(mode="before")
    @classmethod
    def process_labels(cls, data: dict) -> dict:
        if isinstance(data, dict):
            # Handle raw probabilities with model version
            if any(isinstance(k, int) for k in data.keys()):
                return {
                    **{LABEL_CLASS_TO_NAME[k]: v for k, v in data.items() if isinstance(k, int)},
                    "model_version": data.get("model_version", "unknown")
                }
            return data
        raise ValueError(f"Invalid data format: {data}")

class SimpleModelResponse(BaseModel):
    label: SentimentLabel
    score: float
    model_version: str

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="before")
    @classmethod
    def find_highest_score(cls, data: SimpleModelResults | dict) -> dict:
        if isinstance(data, SimpleModelResults):
            data = data.model_dump()

        scores = {
            k: v for k, v in data.items()
            if k in ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        }
        highest_label = max(scores.items(), key=lambda x: x[1])[0]

        return {
            "label": highest_label,
            "score": scores[highest_label],
            "model_version": data.get("model_version", "unknown")
        }
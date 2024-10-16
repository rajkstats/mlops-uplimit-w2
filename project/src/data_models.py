from pydantic import BaseModel, ConfigDict, model_validator

from src.constants import LABEL_CLASS_TO_NAME, SentimentLabel

# NOTE: If you're using a different model ensure that you add in the Results and ModelResponse
# Pydantic models below!

# Ensures that the API receives a review as a string in the correct format
class SimpleModelRequest(BaseModel):
    review: str

# This model stores the probabilities for each sentiment label: NEGATIVE, NEUTRAL, POSITIVE
# process labels converts raw prediction results {0: 0.1, 1: 0.8, 2: 0.2}. to human readable string {"NEGATIVE": 0.8, "NEUTRAL": 0.1, "POSITIVE": 0.1} using labels
class SimpleModelResults(BaseModel):
    NEGATIVE: float
    NEUTRAL: float
    POSITIVE: float

    @model_validator(mode="before")
    @classmethod
    def process_labels(cls, data: dict[int, float]) -> dict[str, float]:
        return {LABEL_CLASS_TO_NAME[key]: value for key, value in data.items()}

# This model returns the final prediction with the label (e.g., "POSITIVE") and the corresponding probability score
# find_highest_score(): Extracts the label with the highest probability score from the raw prediction results
# Example: {"NEGATIVE": 0.8, "NEUTRAL": 0.1, "POSITIVE": 0.1} to {"label": "NEGATIVE", "score": 0.8}

class SimpleModelResponse(BaseModel):
    label: SentimentLabel
    score: float

    model_config = ConfigDict(use_enum_values=True)

    @model_validator(mode="before")
    @classmethod
    def find_highest_score(
        cls, data: dict[str, float]
    ) -> dict[
        str,
        float | str,
    ]:
        highest_label, highest_score = max(
            data.items(),
            key=lambda item: item[1],
        )
        return {"label": highest_label, "score": highest_score}

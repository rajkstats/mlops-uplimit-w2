import numpy as np
import onnxruntime as rt
import wandb
import os
from typing import Literal, Dict

from src.constants import (
    WANDB_API_KEY,
    OLD_MODEL_NAME,
    NEW_MODEL_NAME
)


class Model:
    @classmethod
    def load_model(cls, model_version: Literal["old", "new"] = "old") -> rt.InferenceSession:
        if WANDB_API_KEY is None:
            raise ValueError(
                "WANDB_API_KEY not set, unable to pull the model!",
            )

        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
        model_name = OLD_MODEL_NAME if model_version == "old" else NEW_MODEL_NAME

        run = wandb.init(
            project="Drug Review MLOps Uplimit",
            name=f"model_serving_{model_version}",
            job_type="inference",
            reinit=True
        )

        downloaded_model_path = run.use_model(
            name=model_name,
        )
        return rt.InferenceSession(
            downloaded_model_path, providers=["CPUExecutionProvider"]
        )

    @classmethod
    def predict(
            cls, session: rt.InferenceSession, review: str
        ) -> dict[int, float]:
            input_name = session.get_inputs()[0].name
            _, probas = session.run(None, {input_name: np.array([[review]])})

            # Just convert raw probabilities to dictionary
            return {i: float(prob) for i, prob in enumerate(probas[0])}
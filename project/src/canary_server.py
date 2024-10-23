import random
from fastapi import FastAPI, Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger
import time
import uuid
from datetime import datetime

from src.canary_data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.constants import CANARY_PERCENT
from src.canary_model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier with Canary Deployment",
    version="0.2",
)

def configure_logger(log_file):
    """Configure loguru logger with rotation"""
    logger_instance = logger
    logger_instance.add(log_file, rotation="1 MB")
    return logger_instance

@app.middleware("http")
async def log_and_inject_metadata(request: Request, call_next):
    """Middleware for request logging and metadata injection"""
    request_logger = configure_logger("api.log")
    start_time = time.time()
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    request_body = await request.body()

    response = await call_next(request)

    latency = time.time() - start_time
    request_logger.info(f"Request ID: {request_id}, Timestamp: {timestamp}, Latency: {latency * 1000:.2f}ms, Input: {request_body.decode('utf-8')}")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Timestamp"] = timestamp
    response.headers["X-Latency-ms"] = f"{latency * 1000:.2f}"
    return response

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2, "memory": 512 * 1024 * 1024},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class SimpleModel:
    def __init__(self, model_version: str = "english_v1") -> None:
        self.logger = configure_logger("model.log")
        self.session = Model.load_model(
            "old" if model_version == "english_v1" else "new"
        )
        self.model_version = model_version
        self.logger.info(f"SimpleModel initialized with version: {model_version}")

    def predict(self, review: str) -> SimpleModelResults:
        self.logger.info(f"[{self.model_version}] Predicting sentiment for review: {review}")
        try:
            # Get prediction from model
            raw_probs = Model.predict(self.session, review)

            # Add model version to raw probabilities
            raw_probs["model_version"] = self.model_version

            # SimpleModelResults will process the raw probabilities using its validator
            validated_result = SimpleModelResults.model_validate(raw_probs)
            self.logger.info(f"[{self.model_version}] Prediction result: {validated_result}")
            return validated_result

        except Exception as e:
            self.logger.error(f"[{self.model_version}] Error during prediction: {str(e)}")
            raise

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class Canary:
    """Canary deployment handler for routing between old and new models"""
    def __init__(self, old_model: DeploymentHandle, new_model: DeploymentHandle, canary_percent: float):
        self.logger = configure_logger("canary.log")
        self.old_model = old_model
        self.new_model = new_model
        self.canary_percent = canary_percent
        self.request_count = 0
        self.canary_count = 0
        self.logger.info(f"Canary initialized with {canary_percent*100}% traffic to new model")

    async def predict(self, request: SimpleModelRequest) -> SimpleModelResponse:
        """Route requests between models based on canary percentage"""
        self.request_count += 1
        use_new_model = random.random() < self.canary_percent

        try:
            if use_new_model:
                self.canary_count += 1
                self.logger.info(f"Request {self.request_count} routed to new model")
                model_results = await self.new_model.predict.remote(request.review)
            else:
                self.logger.info(f"Request {self.request_count} routed to old model")
                model_results = await self.old_model.predict.remote(request.review)

            # Log routing statistics
            if self.request_count % 10 == 0:
                current_percentage = (self.canary_count/self.request_count)*100
                self.logger.info(
                    f"Canary Stats - Total: {self.request_count}, "
                    f"New Model: {self.canary_count} ({current_percentage:.1f}%)"
                )

            # Convert to response using SimpleModelResponse validator
            return SimpleModelResponse.model_validate(model_results)

        except Exception as e:
            self.logger.error(f"Error in canary routing: {str(e)}")
            raise

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class APIIngress:
    """API endpoint handling canary deployment"""
    def __init__(self, canary_handle: DeploymentHandle) -> None:
        self.logger = configure_logger("api.log")
        self.logger.info("APIIngress initialized with canary routing")
        self.handle = canary_handle

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest):
        self.logger.info(f"Received prediction request: {request}")
        try:
            result = await self.handle.predict.remote(request)
            self.logger.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

old_model = SimpleModel.bind(model_version="english_v1")
new_model = SimpleModel.bind(model_version="french_v1")
canary = Canary.bind(old_model, new_model, canary_percent=CANARY_PERCENT)
entrypoint = APIIngress.bind(canary)
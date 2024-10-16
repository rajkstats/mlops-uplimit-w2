from fastapi import FastAPI , Request
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger

import time
import uuid
from datetime import datetime
from src.data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier",
    version="0.1",
)

# Add in appropriate logging using loguru wherever you see fit in order to aid with debugging issues.

# Custom function to configure logger dynamically within each request or class
def configure_logger(log_file):
    logger_instance = logger
    logger_instance.add(log_file, rotation="1 MB")
    return logger_instance

# Custom Middleware for logging and injecting metadata
@app.middleware("http")
async def log_and_inject_metadata(request: Request, call_next):
    # Dynamically configure a logger for the request cycle
    request_logger = configure_logger("api.log")

    start_time = time.time()  # Track request start time for latency calculation
    request_id = str(uuid.uuid4())  # Generate a unique request ID
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp

    request_body = await request.body()  # Read request body to log input

    # Calculate latency and log enriched data
    response = await call_next(request)
    latency = time.time() - start_time

    request_logger.info(f"Request ID: {request_id}, Timestamp: {timestamp}, Latency: {latency * 1000:.2f}ms, Input: {request_body.decode('utf-8')}")

    # Inject metadata into response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Timestamp"] = timestamp
    response.headers["X-Latency-ms"] = f"{latency * 1000:.2f}"

    return response



@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, simple_model_handle: DeploymentHandle) -> None:
        # Dynamically configure logger inside the constructor
        self.logger = configure_logger("api.log")
        self.logger.info("APIIngress initialized")
        self.handle = simple_model_handle

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest):
        self.logger.info(f"Received prediction request: {request}")
        try:
            result = await self.handle.predict.remote(request.review)
            self.logger.info(f"Prediction result: {result}")
            return SimpleModelResponse.model_validate(result.model_dump())
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise


@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)

class SimpleModel:
    def __init__(self) -> None:
        # Dynamically configure logger inside the constructor
        self.logger = configure_logger("model.log")
        self.session = Model.load_model()
        self.logger.info("SimpleModel initialized")

    def predict(self, review: str) -> SimpleModelResults:
        self.logger.info(f"Predicting sentiment for review: {review}")
        # Use the Model.predict to get the result
        try:
            result = Model.predict(self.session, review)
            self.logger.info(f"Prediction result: {result}")
            return SimpleModelResults.model_validate(result)
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

entrypoint = APIIngress.bind(
    SimpleModel.bind(),
)

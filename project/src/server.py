from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle
from loguru import logger


from src.data_models import SimpleModelRequest, SimpleModelResponse, SimpleModelResults
from src.model import Model

app = FastAPI(
    title="Drug Review Sentiment Analysis",
    description="Drug Review Sentiment Classifier",
    version="0.1",
)

# Add in appropriate logging using loguru wherever you see fit in order to aid with debugging issues.

# Configure Loguru logger
logger.add("api.log", rotation="1 MB")

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
@serve.ingress(app)
class APIIngress:
    def __init__(self, simple_model_handle: DeploymentHandle) -> None:
        logger.info("APIIngress initialized")
        self.handle = simple_model_handle

    @app.post("/predict")
    async def predict(self, request: SimpleModelRequest):
        logger.info(f"Received prediction request: {request}")
        try:
            result = await self.handle.predict.remote(request.review)
            logger.info(f"Prediction result: {result}")
            return SimpleModelResponse.model_validate(result.model_dump())
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    async def predict(self, request: SimpleModelRequest):
        logger.info(f"Received prediction request: {request}")
        # Use the handle.predict which is a remote function to get the result
        try:
            result = await self.handle.predict.remote(request.review)
            logger.info(f"Prediction result: {result}")
            return SimpleModelResponse.model_validate(result.model_dump())
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

@serve.deployment(
    ray_actor_options={"num_cpus": 0.2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class SimpleModel:
    def __init__(self) -> None:
        self.session = Model.load_model()
        logger.info("SimpleModel initialized")

    def predict(self, review: str) -> SimpleModelResults:
        logger.info(f"Predicting sentiment for review: {review}")
        # Use the Model.predict to get the result
        try:
            result = Model.predict(self.session, review)
            logger.info(f"Prediction result: {result}")
            return SimpleModelResults.model_validate(result)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

entrypoint = APIIngress.bind(
    SimpleModel.bind(),
)

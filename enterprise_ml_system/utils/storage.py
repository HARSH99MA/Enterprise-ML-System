from pymongo import MongoClient
from typing import Dict, Any
import logging
from models.base_models import ProcessingResult, ModelRegistry

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages data storage operations."""

    def __init__(self, config: Dict[str, Any]):
        self.client = MongoClient(config['mongodb_uri'])
        self.db = self.client[config['database']]
        self.collections = config['collections']

    async def store_result(self, result: ProcessingResult):
        """Store processing result in database."""
        try:
            self.db[self.collections['results']].insert_one(result.dict())
            logger.info(f"Result stored for request: {result.request_hash}")

        except Exception as e:
            logger.error(f"Failed to store result: {e}")
            raise

    async def store_model(self, model_data: ModelRegistry):
        """Store model registry data in database."""
        try:
            self.db[self.collections['models']].insert_one(model_data.dict())
            logger.info(f"Model stored with id: {model_data.model_id}")

        except Exception as e:
            logger.error(f"Failed to store model: {e}")
            raise

    async def store_event(self, event: Dict[str, Any]):
        """Store event in database."""
        try:
            self.db[self.collections['events']].insert_one(event)
            logger.info(f"Event stored: {event['type']}")
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            raise
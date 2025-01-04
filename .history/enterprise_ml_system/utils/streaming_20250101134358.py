from typing import Dict, Any
from kafka import KafkaProducer, KafkaConsumer
import json
import logging

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Handles streaming data using Kafka."""

    def __init__(self, bootstrap_servers: str, topics: Dict[str, str]):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topics = topics

    async def send_event(self, event: Dict[str, Any]):
        """Send event to Kafka topic."""
        try:
            self.producer.send(self.topics['events'], event)
            self.producer.flush()
        except Exception as e:
            logger.error(f"Kafka event send failed: {e}")
            raise

    def consume_messages(self):
     """Consumes messages from the input topic and yields them."""
     consumer = KafkaConsumer(
         self.topics['input'],
         bootstrap_servers=self.bootstrap_servers,
         auto_offset_reset='earliest',
         enable_auto_commit=True,
         group_id='my-group',
         value_deserializer=lambda x: json.loads(x.decode('utf-8'))
     )

     try:
        for message in consumer:
            yield message.value
     except Exception as e:
        logger.error(f"Kafka consumption failed: {e}")
        raise
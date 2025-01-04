import tensorflow as tf
import tensorflow_federated as tff
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class FederatedLearningManager:
    """Manages federated learning tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.num_clients = config['num_clients']
        self.rounds = config['rounds']
        self.local_epochs = config['local_epochs']
        self.model = None
        self.training_data = {}
        self.client_ids = [f'client_{i}' for i in range(self.num_clients)]
        self.federated_process = None

    def create_federated_model(self):
        # Define the model
        def create_keras_model():
            return tf.keras.models.Sequential([
                tf.keras.layers.Dense(10, input_shape=(784,), activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid'),
            ])

        def model_fn():
            keras_model = create_keras_model()
            return tff.learning.from_keras_model(
                keras_model,
                input_spec=tf.TensorSpec(shape=(None, 784), dtype=tf.float32),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
            )

        # Federated Averaging algorithm
        self.federated_process = tff.learning.build_federated_averaging_process(
           model_fn=model_fn,
           client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.02),
        )
        #Initialize the federated model
        self.model = self.federated_process.initialize()
        
    def prepare_client_data(self, client_id: str, data: List[Dict[str, Any]]):
        """Prepare client data for federated training."""
        # Convert the input list of dictionaries to tensors for tensorflow federated
        try:
            if not isinstance(data, list):
                logger.error(f"Invalid training data format for {client_id}.")
                return False
            if not data:
                logger.warning(f"No training data provided for client {client_id}.")
                self.training_data[client_id] = None
                return False
            
            # Check if all items have 'x' and 'y'
            if not all('x' in item and 'y' in item for item in data):
                logger.error(f"Missing 'x' or 'y' in data for client {client_id}.")
                return False
            
            # Separate features and labels and convert to tensors
            client_x = [item['x'] for item in data]
            client_y = [item['y'] for item in data]

            client_x = tf.convert_to_tensor(client_x, dtype=tf.float32)
            client_y = tf.convert_to_tensor(client_y, dtype=tf.int32)

            # Reshape labels if it is a 1D array
            if len(client_y.shape) == 1:
                client_y = tf.reshape(client_y, (-1, 1))

            self.training_data[client_id] = tf.data.Dataset.from_tensor_slices(
                 (client_x, client_y)
            ).batch(32)

            return True

        except Exception as e:
            logger.error(f"Data preparation failed for client {client_id}: {e}")
            return False

    async def process_client_update(self, client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process client update using federated learning."""
        try:
            if self.federated_process is None:
                self.create_federated_model()
            # Prepare client data
            if not self.prepare_client_data(client_id, data.get('federated_data', [])):
                return {
                   'status': 'failed',
                   'message': f'Data preparation failed for {client_id}'
                }

            # If no training data skip training
            if self.training_data.get(client_id) is None:
                return {
                    'status': 'success',
                    'message': f"No training data for client {client_id}."
                }

            # perform training if data is valid
            client_datasets = [self.training_data[client_id] for _ in range(self.rounds) if self.training_data.get(client_id) is not None]
            client_datasets_dict = {client_id: ds for ds in client_datasets}

            client_updates = {client_id: client_datasets_dict[client_id] for client_id in self.client_ids if client_datasets_dict.get(client_id) is not None}

            if not client_updates:
                return {
                   'status': 'failed',
                   'message': "No valid client datasets for federated learning."
                }

            # Run federated learning
            for i in range(self.rounds):
               self.model = self.federated_process.next(self.model, list(client_updates.values()))

            # Retrieve trained model weights
            trained_weights = self.federated_process.get_model_weights(self.model)
            return {
               'status': 'success',
               'message': f'Federated learning completed for client {client_id}',
               'trained_weights': trained_weights
            }

        except Exception as e:
            logger.error(f"Federated learning process failed: {e}")
            return {
               'status': 'failed',
               'message': f'Federated learning process failed: {e}'
            }
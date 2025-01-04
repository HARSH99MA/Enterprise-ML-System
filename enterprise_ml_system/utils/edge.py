from typing import Dict, Any, List
import logging
import json
import asyncio
import aiohttp
import os
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class EdgeComputingManager:
    """Handles edge computing tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.model_path = config['model_path']
        self.platforms = config['platforms']
        self.quantize = config['quantize']
        self.model = None
        # Preload model
        try:
           self.load_model()
        except Exception as e:
            logger.error(f"Model loading failed {e}")
            raise

    def load_model(self):
        """Load and quantize model."""
        try:
            if os.path.exists(self.model_path + '/saved_model.pb'):
                 self.model = tf.saved_model.load(self.model_path)
                 if self.quantize:
                   self.model = self.quantize_model(self.model)
            else:
                logger.warning(f"Model not found in: {self.model_path} \
                                Assuming model is being loaded on demand by edge device.")

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def quantize_model(self, model):
        """Apply post-training quantization."""
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_tflite_model = converter.convert()
            # Create an interpreter with the quantized model
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
            return interpreter
        except Exception as e:
            logger.error(f"Quantization failed {e}")
            raise

    async def get_prediction(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from edge model."""
        try:
            # Load the model dynamically if it wasn't preloaded
            if self.model is None and not os.path.exists(self.model_path + '/saved_model.pb'):
               logger.warning("No preloaded model. Request will be forwarded to edge device.")

            if device_id not in self.platforms:
                logger.warning(f"Edge device {device_id} is not in supported devices")
                return {
                    'status': 'failed',
                    'message': f'Device {device_id} not supported'
                }

            if self.model is not None:
                 # Preprocess data
                 input_data = np.array([data.get('x', [])], dtype=np.float32)

                 if self.quantize:
                     # Inference with quantized model
                     self.model.allocate_tensors()
                     input_details = self.model.get_input_details()
                     output_details = self.model.get_output_details()
                     self.model.set_tensor(input_details[0]['index'], input_data)
                     self.model.invoke()
                     output_data = self.model.get_tensor(output_details[0]['index'])

                 else:
                     # Inference with non-quantized model
                     output_data = self.model(input_data).numpy()

                 return {
                    'status': 'success',
                    'device_id': device_id,
                    'prediction': output_data.tolist()
                 }

            else:
                # Forward request to device if no model present in cloud
                return await self.send_request_to_edge_device(device_id, data)

        except Exception as e:
            logger.error(f"Edge prediction failed: {e}")
            return {
                'status': 'failed',
                'message': f'Edge prediction failed {e}'
            }

    async def send_request_to_edge_device(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to edge device."""
        try:
            # Simulate sending request to edge device.  In real use case it would be http call or other communication protocol
            logger.info(f"Sending request to edge device {device_id}.")
            # Simulate API call
            async with aiohttp.ClientSession() as session:
                 async with session.post(f'http://{device_id}:8080/predict', json=data) as response:
                    if response.status == 200:
                         result = await response.json()
                         return {
                             'status': 'success',
                             'device_id': device_id,
                             'prediction': result.get('prediction', 'no prediction')
                        }
                    else:
                       return {
                         'status': 'failed',
                         'message': f'Edge prediction failed {response.status}'
                     }
        except Exception as e:
          logger.error(f"Request to edge device failed {e}")
          return {
             'status': 'failed',
             'message': f'Request to edge device failed {e}'
          }
from typing import Dict, Any, List, Callable
from google.cloud import aiplatform
import logging
import os

logger = logging.getLogger(__name__)

class ServerlessFunction:
    """Handles serverless function deployment and execution."""

    def __init__(self, project_id: str, location: str):
        self.client = aiplatform.gapic.PipelineServiceClient(
            client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
        )
        self.project_id = project_id
        self.location = location
        self.bucket_name = os.getenv('BUCKET_NAME')
        
    async def deploy_function(
        self, function_name: str, handler: Callable, requirements: List[str]
    ) -> str:
        """Deploy function to cloud platform."""
        try:
             parent = f"projects/{self.project_id}/locations/{self.location}"

             # Create a unique package URI for each deployment
             package_uri = f"gs://{self.bucket_name}/{function_name}/{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"

             function = {
                 "display_name": function_name,
                 "python_package_spec": {
                     "executor_image_uri": "python:3.9",
                     "package_uris": [package_uri],
                     "requirements": requirements,
                 },
                 "predict_schemata": {
                     "instance_schema_uri": "gs://google-cloud-aiplatform/schema/predict/instance/custom_prediction_io_1.0.0.yaml",
                     "parameters_schema_uri": "gs://google-cloud-aiplatform/schema/predict/params/custom_prediction_io_1.0.0.yaml",
                     "prediction_schema_uri": "gs://google-cloud-aiplatform/schema/predict/prediction/custom_prediction_io_1.0.0.yaml"
                 }
             }
             # Simulate upload of the user code to Google storage bucket
             # with zip_file which will be created from handler function and requirements
             zip_file = self._package_function(handler, requirements)
             # Upload zip file to google storage bucket
             self._upload_blob(package_uri, zip_file)

             response = await self.client.create_model(
                 request={"parent": parent, "model": function}
             )

             return response.result()
        except Exception as e:
             logger.error(f"Function deployment failed: {e}")
             raise

    def _package_function(self, handler, requirements):
        import zipfile
        import inspect
        from io import BytesIO

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:

          # Add main handler function
          handler_src = inspect.getsource(handler)
          zf.writestr('main.py', handler_src)

         # Add requirements to be installed
          zf.writestr('requirements.txt', '\n'.join(requirements))
          # include dependencies if needed

        buffer.seek(0)
        return buffer

    def _upload_blob(self, package_uri, zip_file):
        """Uploads a zip archive to Google Cloud Storage."""
        from google.cloud import storage

        try:
             storage_client = storage.Client()
             bucket_name = package_uri.replace('gs://', '').split('/')[0]
             blob_name = package_uri.replace(f'gs://{bucket_name}/', '')
             bucket = storage_client.bucket(bucket_name)
             blob = bucket.blob(blob_name)
             blob.upload_from_file(zip_file)
             logger.info(f"File {blob_name} uploaded to {bucket_name}.")
        except Exception as e:
           logger.error(f"Failed to upload model to cloud storage: {e}")
           raise
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from datetime import datetime

class ProcessingRequest(BaseModel):
    text: Optional[str]
    nlp_tasks: Optional[List[str]]
    relationships: Optional[List[Dict[str, str]]]
    edge_device_id: Optional[str]
    client_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

class ProcessingResult(BaseModel):
    request_hash: str
    blockchain_tx: str
    nlp_results: Optional[Dict[str, Any]]
    edge_results: Optional[Dict[str, Any]]
    graph_results: Optional[Dict[str, Any]]
    federated_results: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: datetime

class ModelRegistry(BaseModel):
    model_id: str
    version: str
    type: str
    location: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
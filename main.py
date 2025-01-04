   # in main.py
   from dotenv import load_dotenv
   load_dotenv()

import os
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from models.base_models import ProcessingRequest, ProcessingResult
from utils.blockchain import BlockchainManager
from utils.nlp import AdvancedNLPPipeline
from utils.streaming import StreamProcessor
from utils.graph import GraphNeuralNetwork
from utils.federated import FederatedLearningManager
from utils.edge import EdgeComputingManager
from utils.cloud import ServerlessFunction
from utils.storage import StorageManager
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

class EnterpriseMLSystem:
    """Enterprise ML system integrating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.blockchain = BlockchainManager(
            config['blockchain']['uri'],
            config['blockchain']['contract_address'],
            config['blockchain']['contract_abi']
        )
        
        self.nlp_pipeline = AdvancedNLPPipeline(
            config['nlp']
        )
        
        self.stream_processor = StreamProcessor(
            config['streaming']['kafka_bootstrap_servers'],
            config['streaming']['kafka_topics']
        )
        
        self.graph_nn = GraphNeuralNetwork(
            config['graph']['input_dim'],
            config['graph']['hidden_dim'],
            config['graph']['output_dim']
        )
        
        self.federated = FederatedLearningManager(
            config['federated']
        )
        
        self.edge_computing = EdgeComputingManager(
            config['edge']
        )
        
        self.serverless = ServerlessFunction(
            
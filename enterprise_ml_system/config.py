import os

config = {
    'blockchain': {
        'uri': 'http://localhost:8545',
        'contract_address': '0x...',
        'contract_abi': [...],
        'private_key': os.getenv('ETHEREUM_PRIVATE_KEY')
    },
    'nlp': {
        'base_model': 'bert-base-uncased',
        'summarization_model': 't5-small',
        'qa_model': 'distilbert-base-cased-distilled-squad',
        'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english'
    },
    'streaming': {
        'kafka_bootstrap_servers': 'localhost:9092',
        'kafka_topics': {
            'input': 'ml_system_input',
            'output': 'ml_system_output',
            'events': 'ml_system_events'
        }
    },
    'graph': {
        'input_dim': 128,
        'hidden_dim': 64,
        'output_dim': 32,
        'learning_rate': 0.01
    },
    'federated': {
        'num_clients': 5,
        'rounds': 10,
        'local_epochs': 5
    },
    'edge': {
        'model_path': 'models/edge_model',
        'platforms': ['nvidia_jetson', 'raspberry_pi', 'android'],
        'quantize': True
    },
    'cloud': {
        'project_id': 'your-project-id',
        'location': 'us-central1',
        'bucket': 'your-bucket'
    },
    'storage': {
        'mongodb_uri': 'mongodb://localhost:27017/',
        'database': 'ml_system',
        'collections': {
            'results': 'processing_results',
            'models': 'model_registry',
            'events': 'system_events'
        }
    }
}
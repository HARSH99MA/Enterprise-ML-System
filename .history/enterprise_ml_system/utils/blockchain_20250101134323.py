import os
from typing import Dict, List, Any
import json
from web3 import Web3
import logging

logger = logging.getLogger(__name__)

class BlockchainManager:
    """Manages blockchain interactions for data integrity and smart contracts."""

    def __init__(self, blockchain_uri: str, contract_address: str, abi: List[Dict]):
        self.web3 = Web3(Web3.HTTPProvider(blockchain_uri))
        self.contract = self.web3.eth.contract(address=contract_address, abi=abi)
        self.account = self.web3.eth.account.from_key(os.getenv('ETHEREUM_PRIVATE_KEY'))

    def calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculates the keccak-256 hash of the data."""
        return self.web3.keccak(text=json.dumps(data)).hex()

    async def store_hash(self, data_hash: str, metadata: Dict[str, Any]) -> str:
        """Store data hash in blockchain."""
        try:
            # Create transaction
            transaction = self.contract.functions.storeHash(
                data_hash,
                json.dumps(metadata)
            ).build_transaction({
                'from': self.account.address,
                'gas': 2000000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': self.web3.eth.get_transaction_count(self.account.address),
            })

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction, private_key=os.getenv('ETHEREUM_PRIVATE_KEY')
            )
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            return self.web3.to_hex(tx_hash)

        except Exception as e:
            logger.error(f"Blockchain transaction failed: {e}")
            raise
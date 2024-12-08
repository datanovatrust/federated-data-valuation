# src/utils/blockchain_utils.py

import os
import json
import logging
from web3 import Web3
from eth_utils import to_checksum_address  # Add this import

logger = logging.getLogger(__name__)

class BlockchainClient:
    def __init__(self, rpc_url, contract_address, abi_path, private_key=None):
        self.web3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.web3.is_connected():
            logger.error(f"Unable to connect to Ethereum node at {rpc_url}")
            raise ConnectionError("Ethereum node not reachable.")

        with open(abi_path, 'r') as f:
            abi = json.load(f)

        # Use the to_checksum_address function instead of Web3.toChecksumAddress
        checksummed_address = to_checksum_address(contract_address)
        self.contract = self.web3.eth.contract(address=checksummed_address, abi=abi)

        self.private_key = private_key
        if self.private_key:
            self.account = self.web3.eth.account.privateKeyToAccount(self.private_key)
        else:
            self.account = None

    def _build_and_send_tx(self, tx):
        """Build and send a transaction if private_key is provided. Otherwise, raise error."""
        if not self.private_key:
            logger.error("No private key provided. Cannot send transactions.")
            raise PermissionError("No private key to sign transactions with.")

        # Set nonce
        tx['nonce'] = self.web3.eth.get_transaction_count(self.account.address)

        # Sign and send transaction
        tx_signed = self.account.sign_transaction(tx)
        tx_hash = self.web3.eth.send_raw_transaction(tx_signed.rawTransaction)
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
        logger.info(f"Transaction successful with hash: {tx_hash.hex()}")
        return receipt

    def register_participant(self, participant_address):
        """Register a participant on-chain. OnlyOwner restricted."""
        func = self.contract.functions.registerParticipant(Web3.toChecksumAddress(participant_address))
        tx = func.buildTransaction({
            'from': self.account.address,
            'gas': 3000000,
            # Use EIP-1559 style parameters:
            'maxFeePerGas': self.web3.toWei('2', 'gwei'),
            'maxPriorityFeePerGas': self.web3.toWei('1', 'gwei'),
            'chainId': 31337  # Anvil’s default chain ID
        })
        return self._build_and_send_tx(tx)

    def record_update(self, round_num, model_hash):
        """Record a global model hash on-chain."""
        func = self.contract.functions.recordModelHash(round_num, model_hash)
        tx = func.buildTransaction({
            'from': self.account.address,
            'gas': 3000000,
            'maxFeePerGas': self.web3.toWei('2', 'gwei'),
            'maxPriorityFeePerGas': self.web3.toWei('1', 'gwei'),
            'chainId': 31337
        })
        return self._build_and_send_tx(tx)

    def incentivize_participants(self, round_num, participant_list):
        """Distribute incentives to participants."""
        func = self.contract.functions.incentivizeParticipants(round_num, participant_list)
        tx = func.buildTransaction({
            'from': self.account.address,
            'gas': 3000000,
            'maxFeePerGas': self.web3.toWei('2', 'gwei'),
            'maxPriorityFeePerGas': self.web3.toWei('1', 'gwei'),
            'chainId': 31337
        })
        return self._build_and_send_tx(tx)

    def get_model_hash(self, round_num):
        """Retrieve the model hash from the contract (read-only)."""
        # No transaction needed for read calls, so just call the contract function directly
        return self.contract.functions.getModelHash(round_num).call()

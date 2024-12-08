import os
import requests
import logging
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'blockchain_config.yaml')

def load_ipfs_gateway_from_config(config_path=CONFIG_PATH):
    """
    Load the IPFS gateway URL from the blockchain configuration file.
    """
    if not os.path.isfile(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    blockchain_config = config.get('blockchain', {})
    ipfs_gateway = blockchain_config.get('ipfs_gateway')
    if not ipfs_gateway:
        logger.error("IPFS gateway not specified in configuration.")
        raise ValueError("IPFS gateway not found in blockchain_config.yaml")

    return ipfs_gateway

def upload_to_ipfs(file_path):
    """
    Uploads a file to IPFS via the gateway specified in the blockchain configuration
    and returns the content hash.
    """
    ipfs_gateway = load_ipfs_gateway_from_config()

    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    try:
        logger.info(f"Uploading {file_path} to IPFS via {ipfs_gateway}...")
        with open(file_path, 'rb') as f:
            response = requests.post(f"{ipfs_gateway}/api/v0/add", files={"file": f})

        response.raise_for_status()
        ipfs_hash = response.json()['Hash']
        logger.info(f"Successfully uploaded {file_path} to IPFS. Hash: {ipfs_hash}")
        return ipfs_hash
    except requests.RequestException as re:
        logger.error(f"Request error during IPFS upload: {re}")
        raise
    except Exception as e:
        logger.error(f"Failed to upload file to IPFS: {e}")
        raise

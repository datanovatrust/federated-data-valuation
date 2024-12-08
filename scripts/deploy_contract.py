# scripts/deploy_contract.py

from brownie import FLRegistry, accounts, network

def main():
    acct = accounts[0]
    deployed = FLRegistry.deploy(
        {
            'from': acct,
            'max_fee': "2 gwei",
            'priority_fee': "1 gwei"
        }
    )
    print("Contract deployed at:", deployed.address)
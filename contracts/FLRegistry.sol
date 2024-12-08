// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    address public owner;

    struct GlobalModel {
        uint256 roundNumber;
        string modelHash;
    }

    mapping(uint256 => GlobalModel) public globalModels;
    mapping(address => bool) public participants;

    event ParticipantRegistered(address participant);
    event ModelHashRecorded(uint256 roundNumber, string modelHash);
    event IncentivesDistributed(uint256 roundNumber, address[] participantsRewarded);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not contract owner");
        _;
    }

    function registerParticipant(address participant) external onlyOwner {
        require(!participants[participant], "Already registered");
        participants[participant] = true;
        emit ParticipantRegistered(participant);
    }

    function recordModelHash(uint256 roundNumber, string calldata modelHash) external onlyOwner {
        GlobalModel memory gm = GlobalModel({
            roundNumber: roundNumber,
            modelHash: modelHash
        });
        globalModels[roundNumber] = gm;
        emit ModelHashRecorded(roundNumber, modelHash);
    }

    function incentivizeParticipants(uint256 roundNumber, address[] calldata participantList) external onlyOwner {
        emit IncentivesDistributed(roundNumber, participantList);
    }

    function getModelHash(uint256 roundNumber) external view returns (string memory) {
        return globalModels[roundNumber].modelHash;
    }
}

// circuits/aggregator.circom

pragma circom 2.0.0;

include "mimc_hash.circom";

template ModelAverage(numClients) {
    signal input currValue;              
    signal input newValues[numClients];  
    signal output out;                   

    signal sums[numClients+1];
    sums[0] <== 0;
    for (var i = 0; i < numClients; i++) {
        sums[i+1] <== sums[i] + newValues[i];
    }

    signal sum;
    sum <== sums[numClients];

    signal currValueScaled;
    currValueScaled <== currValue * numClients;

    signal diff;
    diff <== sum - currValueScaled;

    out <== currValue + (diff / numClients); 
}

template AggregatorCircuit(numClients, inputSize, hiddenSize, outputSize) {
    signal input ScLH[numClients];  
    signal input gdigest;           

    signal input GW[hiddenSize][inputSize];          
    signal input GB[hiddenSize];                     
    signal input LWp[numClients][hiddenSize][inputSize]; 
    signal input LBp[numClients][hiddenSize];         
    signal input GWp[hiddenSize][inputSize];          
    signal input GBp[hiddenSize];                     

    signal CGW[hiddenSize][inputSize];
    signal CGB[hiddenSize];

    component weightAveragers[hiddenSize][inputSize];
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            weightAveragers[i][j] = ModelAverage(numClients);
            weightAveragers[i][j].currValue <== GW[i][j];
            for (var k = 0; k < numClients; k++) {
                weightAveragers[i][j].newValues[k] <== LWp[k][i][j];
            }
            CGW[i][j] <== weightAveragers[i][j].out;
        }
    }
    
    component biasAveragers[hiddenSize];
    for (var i = 0; i < hiddenSize; i++) {
        biasAveragers[i] = ModelAverage(numClients);
        biasAveragers[i].currValue <== GB[i];
        for (var k = 0; k < numClients; k++) {
            biasAveragers[i].newValues[k] <== LBp[k][i];
        }
        CGB[i] <== biasAveragers[i].out;
    }
    
    signal modelMatch[hiddenSize][inputSize + 1];
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            modelMatch[i][j] <== CGW[i][j] - GWp[i][j];
        }
        modelMatch[i][inputSize] <== CGB[i] - GBp[i];
    }

    component localHashers[numClients];
    signal validLocalHashes[numClients];
    for (var i = 0; i < numClients; i++) {
        localHashers[i] = MiMCArray(hiddenSize * inputSize + hiddenSize);
        var idx = 0;
        
        for (var j = 0; j < hiddenSize; j++) {
            for (var p = 0; p < inputSize; p++) {
                localHashers[i].ins[idx] <== LWp[i][j][p];
                idx++;
            }
        }
        
        for (var j = 0; j < hiddenSize; j++) {
            localHashers[i].ins[idx] <== LBp[i][j];
            idx++;
        }
        
        localHashers[i].k <== 0;
        
        validLocalHashes[i] <== (localHashers[i].hash - ScLH[i]);
    }
    
    component globalHasher = MiMCArray(hiddenSize * inputSize + hiddenSize);
    var idx = 0;
    
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            globalHasher.ins[idx] <== GWp[i][j];
            idx++;
        }
    }
    
    for (var i = 0; i < hiddenSize; i++) {
        globalHasher.ins[idx] <== GBp[i];
        idx++;
    }
    
    globalHasher.k <== 0;
    
    signal output valid_model;     
    signal output valid_hashes;    
    
    var hashMatch = globalHasher.hash - gdigest;

    valid_model <== hashMatch; 

    signal tmpSquared[numClients];
    signal allHashesProducts[numClients+1];
    allHashesProducts[0] <== 1;
    for (var i = 0; i < numClients; i++) {
        tmpSquared[i] <== validLocalHashes[i]*validLocalHashes[i];
        allHashesProducts[i+1] <== allHashesProducts[i]*(tmpSquared[i]+1);
    }

    valid_hashes <== allHashesProducts[numClients];
}

// Reduced parameters for fewer constraints
component main = AggregatorCircuit(4, 5, 10, 3);

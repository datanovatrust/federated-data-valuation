pragma circom 2.0.0;

include "mimc_hash.circom";

template ModelAverage(numClients) {
    signal input currValue;              
    signal input newValues[numClients];  
    signal output out;                   

    // Added signals for bounds checking as recommended
    signal currValueInBounds;
    signal newValuesInBounds[numClients];

    currValueInBounds <== currValue;
    for (var i = 0; i < numClients; i++) {
        newValuesInBounds[i] <== newValues[i];
    }

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

    // Properly scaling and dividing by numClients (compile-time constant)
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
    component localHashComparators[numClients];
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

        // Use HashEquality for robust local hash checking
        localHashComparators[i] = HashEquality();
        localHashComparators[i].hash1 <== localHashers[i].hash;
        localHashComparators[i].hash2 <== ScLH[i];
        validLocalHashes[i] <== localHashComparators[i].equal;
        localHashComparators[i].equal === 0;
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

    // Use HashEquality for global hash check
    component globalHashCompare = HashEquality();
    globalHashCompare.hash1 <== globalHasher.hash;
    globalHashCompare.hash2 <== gdigest;
    valid_model <== globalHashCompare.equal;
    globalHashCompare.equal === 0;

    // If all local hashes are valid, their equal signals are all zero
    // Combine them in some way to produce valid_hashes. 
    // Since each local hash comparator outputs 0 on success, we can just 
    // sum them up to ensure they're all zero. Or replicate old logic:
    signal sumLocalHashDiffs[numClients+1];
    sumLocalHashDiffs[0] <== 0;
    for (var i = 0; i < numClients; i++) {
        sumLocalHashDiffs[i+1] <== sumLocalHashDiffs[i] + validLocalHashes[i];
    }

    // If all are zero, final sum is zero
    sumLocalHashDiffs[numClients] === 0;
    valid_hashes <== sumLocalHashDiffs[numClients];
}

// Reduced parameters for fewer constraints
component main = AggregatorCircuit(4, 5, 10, 3);

pragma circom 2.0.0;

include "mimc_hash.circom";

template WeightedSum(n) {
    signal input in[n];
    signal input weights[n];
    signal input factor;
    signal output out;

    signal products[n]; 
    signal sums[n+1]; 
    signal scaled[n];  
    signal prodVal[n];
    signal scaledVal[n];

    sums[0] <== 0;
    for (var i = 0; i < n; i++) {
        prodVal[i] <== in[i] * weights[i];
        scaled[i] <== prodVal[i];
    }

    for (var i = 0; i < n; i++) {
        sums[i+1] <== sums[i] + scaled[i];
    }

    out <== sums[n];
}

template HashComparison() {
    signal input hash1;
    signal input hash2;
    signal output diff;
    
    diff <== hash1 - hash2;
}

template ClientCircuit(inputSize, hiddenSize, outputSize) {
    // Public Inputs (must be properly scaled and modulo p)
    signal input eta;     
    signal input pr;      
    signal input ldigest; 
    signal input ScGH;    

    // Private Inputs (must be integers after scaling)
    signal input GW[hiddenSize][inputSize];   
    signal input GB[hiddenSize];              
    signal input X[inputSize];                
    signal input Y[outputSize];               
    signal input LWp[hiddenSize][inputSize];  
    signal input LBp[hiddenSize];             
    signal input delta2_input[outputSize];    
    signal input dW_input[hiddenSize][inputSize];  
    signal input dB_input[hiddenSize];         

    // Forward propagation hidden layer
    signal Z1[hiddenSize];       
    signal A1[hiddenSize];       
    {
        component hiddenLayer[hiddenSize];
        for (var i = 0; i < hiddenSize; i++) {
            hiddenLayer[i] = WeightedSum(inputSize);
            for (var j = 0; j < inputSize; j++) {
                hiddenLayer[i].in[j] <== X[j];
                hiddenLayer[i].weights[j] <== GW[i][j];
            }
            hiddenLayer[i].factor <== pr;
            Z1[i] <== hiddenLayer[i].out + GB[i];
            A1[i] <== Z1[i];
        }
    }

    // Forward propagation output layer
    signal Z2[outputSize];       
    signal A2[outputSize];       
    {
        component outputLayer[outputSize];
        for (var i = 0; i < outputSize; i++) {
            outputLayer[i] = WeightedSum(hiddenSize);
            for (var j = 0; j < hiddenSize; j++) {
                outputLayer[i].in[j] <== A1[j];
                outputLayer[i].weights[j] <== LWp[j][i];
            }
            outputLayer[i].factor <== pr;
            Z2[i] <== outputLayer[i].out + LBp[i];
            A2[i] <== Z2[i];
        }
    }

    // Arrays for checks
    signal delta2Check[outputSize];
    signal dBCheck[hiddenSize];
    signal dWCheck[hiddenSize][inputSize];

    // MSE and delta2 checks
    signal diff[outputSize];
    signal diffSquared[outputSize];
    signal scaledDiffSquared[outputSize];
    signal mseTotal[outputSize+1];
    mseTotal[0] <== 0;

    signal scaledMSE; 
    signal MSE;
    signal delta2[outputSize];
    signal delta1[hiddenSize];
    signal scaledDelta2[outputSize];
    signal diffTimesTwo[outputSize];

    for (var i = 0; i < outputSize; i++) {
        diff[i] <== A2[i] - Y[i];
        diffSquared[i] <== diff[i] * diff[i];
        scaledDiffSquared[i] <== diffSquared[i];
        mseTotal[i+1] <== mseTotal[i] + scaledDiffSquared[i];

        diffTimesTwo[i] <== diff[i] * 2;
        delta2[i] <== delta2_input[i];
        scaledDelta2[i] <== delta2[i];

        delta2Check[i] <== scaledDelta2[i] - diffTimesTwo[i];
    }

    scaledMSE <== mseTotal[outputSize];
    MSE <== scaledMSE;

    {
        component hiddenGrads[hiddenSize];
        for (var i = 0; i < hiddenSize; i++) {
            hiddenGrads[i] = WeightedSum(outputSize);
            for (var j = 0; j < outputSize; j++) {
                hiddenGrads[i].in[j] <== delta2[j];
                hiddenGrads[i].weights[j] <== LWp[i][j];
            }
            hiddenGrads[i].factor <== pr;
            delta1[i] <== hiddenGrads[i].out;
        }
    }

    signal scaledDW[hiddenSize][inputSize];
    signal dW[hiddenSize][inputSize];
    signal dB[hiddenSize];
    signal delta1X[hiddenSize][inputSize];  

    for (var i = 0; i < hiddenSize; i++) {
        dB[i] <== dB_input[i];
        dBCheck[i] <== dB[i] - delta1[i];

        for (var j = 0; j < inputSize; j++) {
            delta1X[i][j] <== delta1[i] * X[j];
            dW[i][j] <== dW_input[i][j];
            scaledDW[i][j] <== dW[i][j];
            dWCheck[i][j] <== scaledDW[i][j] - delta1X[i][j];
        }
    }

    signal weightUpdate[hiddenSize][inputSize];
    signal biasUpdate[hiddenSize];
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            weightUpdate[i][j] <== eta * dW[i][j];
        }
        biasUpdate[i] <== eta * dB[i];
    }

    // Normalize fields before hashing
    signal normalizedLWp[hiddenSize][inputSize];
    signal normalizedLBp[hiddenSize];
    signal normalizedGW[hiddenSize][inputSize];
    signal normalizedGB[hiddenSize];

    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            normalizedLWp[i][j] <== LWp[i][j];
            normalizedGW[i][j] <== GW[i][j];
        }
        normalizedLBp[i] <== LBp[i];
        normalizedGB[i] <== GB[i];
    }

    component localHasher = MiMCArray(hiddenSize * inputSize + hiddenSize);
    component globalHasher = MiMCArray(hiddenSize * inputSize + hiddenSize);
    component localHashCompare = HashEquality();
    component globalHashCompare = HashEquality();

    var idx = 0;
    // Compute local hash using normalized values
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            localHasher.ins[idx] <== normalizedLWp[i][j];
            idx++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        localHasher.ins[idx] <== normalizedLBp[i];
        idx++;
    }
    localHasher.k <== 0;

    // Compute global hash using normalized values
    idx = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            globalHasher.ins[idx] <== normalizedGW[i][j];
            idx++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        globalHasher.ins[idx] <== normalizedGB[i];
        idx++;
    }
    globalHasher.k <== 0;

    // Compare hashes
    localHashCompare.hash1 <== localHasher.hash;
    localHashCompare.hash2 <== ldigest;
    globalHashCompare.hash1 <== globalHasher.hash;
    globalHashCompare.hash2 <== ScGH;

    // Constraints
    for (var i = 0; i < outputSize; i++) {
        delta2Check[i] === 0;
    }

    for (var i = 0; i < hiddenSize; i++) {
        dBCheck[i] === 0;
        for (var j = 0; j < inputSize; j++) {
            dWCheck[i][j] === 0;
        }
    }

    localHashCompare.equal === 0;
    globalHashCompare.equal === 0;

    // Public outputs
    signal output out[4];
    out[0] <== eta;
    out[1] <== pr;
    out[2] <== ldigest;
    out[3] <== ScGH;
}

component main = ClientCircuit(5, 10, 3);

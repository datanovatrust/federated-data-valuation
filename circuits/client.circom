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
        scaledVal[i] <== prodVal[i] * factor;
        products[i] <== prodVal[i];
        scaled[i] <== scaledVal[i];
    }
    
    for (var i = 0; i < n; i++) {
        sums[i+1] <== sums[i] + scaled[i];
    }
    
    out <== sums[n];
}

template ClientCircuit(inputSize, hiddenSize, outputSize) {
    // Public Inputs (now also marked as outputs)
    signal input eta;       
    signal input pr;        
    signal input ldigest;   
    signal input ScGH;      
    
    // Make public inputs available as outputs
    signal output out_eta;
    signal output out_pr;
    signal output out_ldigest;
    signal output out_ScGH;

    // Private Inputs
    signal input GW[hiddenSize][inputSize];  
    signal input GB[hiddenSize];             
    signal input X[inputSize];               
    signal input Y[outputSize];              
    signal input LWp[hiddenSize][inputSize]; 
    signal input LBp[hiddenSize];            
    signal input delta2_input[outputSize];   
    signal input dW_input[hiddenSize][inputSize];  
    signal input dB_input[hiddenSize];       

    // Forward propagation
    signal Z1[hiddenSize];       
    signal A1[hiddenSize];       
    signal Z2[outputSize];       
    signal A2[outputSize];       

    // MSE and backprop
    signal diff[outputSize];
    signal diffSquared[outputSize];
    signal scaledDiffSquared[outputSize];
    signal mseTotal[outputSize+1];
    signal scaledMSE;
    signal MSE;

    signal delta2[outputSize];
    signal delta1[hiddenSize];
    signal scaledDelta2[outputSize];
    signal diffTimesTwo[outputSize];

    signal scaledDW[hiddenSize][inputSize];
    signal dW[hiddenSize][inputSize];
    signal dB[hiddenSize];
    signal delta1X[hiddenSize][inputSize];  

    // Hidden layer
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

    // Output layer
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

    // Compute MSE and verify delta2 relationships
    mseTotal[0] <== 0;
    for (var i = 0; i < outputSize; i++) {
        diff[i] <== A2[i] - Y[i];
        diffSquared[i] <== diff[i] * diff[i];
        scaledDiffSquared[i] <== diffSquared[i] * pr;
        mseTotal[i+1] <== mseTotal[i] + scaledDiffSquared[i];
        
        diffTimesTwo[i] <== diff[i] * 2;
        delta2[i] <== delta2_input[i];
        scaledDelta2[i] <== delta2[i] * pr;
        scaledDelta2[i] === diffTimesTwo[i];
    }

    scaledMSE <== mseTotal[outputSize] / outputSize;
    MSE <== scaledMSE;

    // Compute delta1
    component hiddenGrads[hiddenSize];
    for (var i = 0; i < hiddenSize; i++) {
        hiddenGrads[i] = WeightedSum(outputSize);
        for (var j = 0; j < outputSize; j++) {
            hiddenGrads[i].in[j] <== delta2[j];
            hiddenGrads[i].weights[j] <== LWp[i][j];
        }
        hiddenGrads[i].factor <== 1;
        delta1[i] <== hiddenGrads[i].out;
    }

    for (var i = 0; i < hiddenSize; i++) {
        dB[i] <== dB_input[i];
        dB[i] === delta1[i];  
        
        for (var j = 0; j < inputSize; j++) {
            delta1X[i][j] <== delta1[i] * X[j];
            dW[i][j] <== dW_input[i][j];
            scaledDW[i][j] <== dW[i][j] * pr;
            scaledDW[i][j] === delta1X[i][j];
        }
    }

    signal weightUpdatesValid[hiddenSize][inputSize];
    signal biasUpdatesValid[hiddenSize];
    signal weightUpdate[hiddenSize][inputSize];
    signal biasUpdate[hiddenSize];

    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            weightUpdate[i][j] <== eta * dW[i][j];
            weightUpdatesValid[i][j] <== LWp[i][j] - (GW[i][j] - weightUpdate[i][j]);
        }
        biasUpdate[i] <== eta * dB[i];
        biasUpdatesValid[i] <== LBp[i] - (GB[i] - biasUpdate[i]);
    }

    // Local hasher
    component localHasher = MiMCArray(hiddenSize * inputSize + hiddenSize);
    var idx = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            localHasher.ins[idx] <== LWp[i][j];
            idx++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        localHasher.ins[idx] <== LBp[i];
        idx++;
    }
    localHasher.k <== 0;

    // Global hasher
    component globalHasher = MiMCArray(hiddenSize * inputSize + hiddenSize);
    idx = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            globalHasher.ins[idx] <== GW[i][j];
            idx++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        globalHasher.ins[idx] <== GB[i];
        idx++;
    }
    globalHasher.k <== 0;

    signal localHashMatch;
    signal globalHashMatch;

    localHashMatch <== localHasher.hash - ldigest;
    globalHashMatch <== globalHasher.hash - ScGH;

    // Connect input signals to output signals
    out_eta <== eta;
    out_pr <== pr;
    out_ldigest <== ldigest;
    out_ScGH <== ScGH;

    // Original outputs
    signal output valid_computation;
    signal output valid_hashes;
    valid_computation <== localHashMatch; 
    valid_hashes <== globalHashMatch;
}

component main = ClientCircuit(5, 10, 3);
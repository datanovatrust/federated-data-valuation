pragma circom 2.0.0;

include "mimc_hash_v2.circom";

/**
 * Minimal helper for multiplication to avoid uninitialized-signal or 
 * referencing LHS as an expression.
 */
template Mul2() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

/**
 * A "free variable" template to produce an unconstrained signal x.
 */
template SingleVariable() {
    signal output x;
    // No constraints => solver picks x as needed.
}

/**
 * WeightedSum(n):
 *   We do partialSum of in[i]*weights[i], then out = partialSum / factor
 *   (but rank-1 form: out * factor == partialSum).
 */
template WeightedSum(n) {
    signal input in[n];
    signal input weights[n];
    signal input factor;
    signal output out;

    // partial sums
    signal product[n];
    signal partialSum[n+1];
    partialSum[0] <== 0;

    for (var i = 0; i < n; i++) {
        product[i] <== in[i] * weights[i];
        partialSum[i+1] <== partialSum[i] + product[i];
    }

    // free variable for final result
    component outVar = SingleVariable();

    // multiply outVar.x by factor
    component mulComp = Mul2();
    signal tmpMul;
    signal checkConstraint;

    mulComp.a <== outVar.x;
    mulComp.b <== factor;
    tmpMul <== mulComp.c;

    // ensure tmpMul == partialSum[n]
    checkConstraint <== tmpMul - partialSum[n];
    checkConstraint === 0;

    // WeightedSum output
    out <== outVar.x;
}

/**
 * ClientCircuit
 *   - WeightedSum for hidden + output layers
 *   - Flatten & hash old global => scgh
 *   - Flatten & hash new local => ldigest
 */
template ClientCircuit(inputSize, hiddenSize, outputSize) {
    //
    // --------------- PUBLIC INPUTS ---------------
    //
    signal input eta;
    signal input pr;
    signal input ldigest;
    signal input scgh;

    //
    // --------------- PRIVATE INPUTS --------------
    //
    signal input GW[hiddenSize][inputSize];
    signal input GB[hiddenSize];
    signal input LWp[hiddenSize][outputSize];
    signal input LBp[outputSize];
    signal input X[inputSize];
    signal input Y[outputSize];

    // Hidden WeightedSum
    component hiddenWS[hiddenSize];
    signal Z1[hiddenSize];
    signal A1[hiddenSize];
    signal hiddenTemp[hiddenSize];

    for (var i = 0; i < hiddenSize; i++) {
        hiddenWS[i] = WeightedSum(inputSize);
        for (var j = 0; j < inputSize; j++) {
            hiddenWS[i].in[j] <== X[j];
            hiddenWS[i].weights[j] <== GW[i][j];
        }
        hiddenWS[i].factor <== pr;
    }

    for (var i = 0; i < hiddenSize; i++) {
        hiddenTemp[i] <== hiddenWS[i].out + GB[i];
        Z1[i] <== hiddenTemp[i];
        A1[i] <== hiddenTemp[i];
    }

    // Output WeightedSum
    component outWS[outputSize];
    signal Z2[outputSize];
    signal A2[outputSize];
    signal outTemp[outputSize];

    for (var o = 0; o < outputSize; o++) {
        outWS[o] = WeightedSum(hiddenSize);
        for (var h = 0; h < hiddenSize; h++) {
            outWS[o].in[h] <== A1[h];
            outWS[o].weights[h] <== LWp[h][o];
        }
        outWS[o].factor <== pr;
    }

    for (var o = 0; o < outputSize; o++) {
        outTemp[o] <== outWS[o].out + LBp[o];
        Z2[o] <== outTemp[o];
        A2[o] <== outTemp[o];
    }

    // optional MSE-like check
    signal diff[outputSize];
    signal delta2[outputSize];
    for (var i = 0; i < outputSize; i++) {
        diff[i] <== A2[i] - Y[i];
        delta2[i] <== diff[i] * 2;
    }

    // Flatten+hash old global => scgh
    var sizeGlobal = (hiddenSize * inputSize) + hiddenSize;
    signal globalFlat[sizeGlobal];
    component globalHasher = MiMCArray(sizeGlobal);

    var gi = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            globalFlat[gi] <== GW[i][j];
            gi++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        globalFlat[gi] <== GB[i];
        gi++;
    }

    for (var idxG = 0; idxG < sizeGlobal; idxG++) {
        globalHasher.ins[idxG] <== globalFlat[idxG];
    }
    globalHasher.k <== 0;

    component globalCompare = HashEquality();
    globalCompare.hash1 <== globalHasher.hash;
    globalCompare.hash2 <== scgh;
    globalCompare.equal === 0;

    // Flatten+hash new local => ldigest
    var sizeLocal = (hiddenSize * outputSize) + outputSize;
    signal localFlat[sizeLocal];
    component localHasher = MiMCArray(sizeLocal);

    var li = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var o = 0; o < outputSize; o++) {
            localFlat[li] <== LWp[i][o];
            li++;
        }
    }
    for (var o = 0; o < outputSize; o++) {
        localFlat[li] <== LBp[o];
        li++;
    }

    for (var idxL = 0; idxL < sizeLocal; idxL++) {
        localHasher.ins[idxL] <== localFlat[idxL];
    }
    localHasher.k <== 0;

    component localCompare = HashEquality();
    localCompare.hash1 <== localHasher.hash;
    localCompare.hash2 <== ldigest;
    localCompare.equal === 0;

    // Output signals
    signal output out[4];
    out[0] <== eta;
    out[1] <== pr;
    out[2] <== ldigest;
    out[3] <== scgh;

    // Debug outputs: differences for scgh, ldigest
    signal output checkGlobalDiff;
    checkGlobalDiff <== globalCompare.hash1 - globalCompare.hash2;

    signal output checkLocalDiff;
    checkLocalDiff <== localCompare.hash1 - localCompare.hash2;
}

// Instantiate
component main = ClientCircuit(5, 10, 3);

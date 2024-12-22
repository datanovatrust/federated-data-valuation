pragma circom 2.0.0;

include "mimc_hash_v2.circom";

/**
 * Minimal helper for multiplication of two signals
 * (to avoid TAC02 issues with expressions on the LHS of `<==`).
 */
template Mul2() {
    signal input a;
    signal input b;
    signal output c;

    c <== a * b;
}

/**
 * A template for a "free variable": an unconstrained signal x 
 * that the solver can pick to satisfy constraints referencing x.
 */
template SingleVariable() {
    signal output x;
    // No constraints => solver picks x as needed
}

/**
 * @title AggregatorCircuit (v2)
 * @notice
 *   - Verifies aggregator used valid local models (ScLH[]).
 *   - Recomputes updated global via FedAvg-like approach, rewriting divisions as multiplications.
 *   - Confirms final global hash = gdigest.
 *
 * Circom 2.0 restrictions:
 *   - T2011: No signals/components declared inside for-loops => we declare them at top scope.
 *   - T3001: No direct division => we do multiplication constraints with Mul2.
 *   - TAC02: No expression on the LHS of `<==`; we pass multiplications through Mul2 outputs.
 *   - All "free" signals must be assigned or come from a component output (or they'd be uninitialized).
 */
template AggregatorCircuit(numClients, inputSize, hiddenSize) {

    //
    // ---------------- PUBLIC INPUTS ----------------
    //
    signal input ScLH[numClients];  // local model hashes
    signal input gdigest;           // final global hash

    //
    // ---------------- PRIVATE INPUTS ---------------
    //
    signal input GW[hiddenSize][inputSize];  // old global weights
    signal input GB[hiddenSize];             // old global biases

    signal input LWp[numClients][hiddenSize][inputSize]; 
    signal input LBp[numClients][hiddenSize];

    signal input GWp[hiddenSize][inputSize]; // new global weights
    signal input GBp[hiddenSize];            // new global biases

    //
    // ======== 1) Weighted average for the Weights ========
    // newWeight = oldWeight + ( (sumLocalW) - (oldWeight * numClients) ) / numClients
    // => define diffW = sumLocalW - oldScaled
    // => deltaW * numClients = diffW
    // => newWeight = oldWeight + deltaW
    //

    // We must pre-declare all signals & components outside loops.

    // partialSum for local W
    signal partialSumW[hiddenSize][inputSize][numClients+1];
    signal sumLocalW[hiddenSize][inputSize];
    signal oldScaledW[hiddenSize][inputSize];
    signal diffW[hiddenSize][inputSize];

    // We'll store the "free variable" for each (i, j) as deltaWVal
    // which is the output of SingleVariable
    component deltaWComp[hiddenSize * inputSize];
    signal deltaWVal[hiddenSize][inputSize];

    // For multiplication constraints: mulCompW, plus tmpMulW & checkCW
    component mulCompW[hiddenSize * inputSize];
    signal tmpMulW[hiddenSize][inputSize];
    signal checkCW[hiddenSize][inputSize];

    // The final computed global weights
    signal computedGW[hiddenSize][inputSize];

    //
    // ======== 2) Weighted average for the Biases ========
    // newBias = oldBias + ( (sumLocalB) - (oldBias * numClients) ) / numClients
    // => define diffB = sumLocalB - oldBscaled
    // => deltaB * numClients = diffB
    // => newBias = oldBias + deltaB
    //

    signal partialSumB[hiddenSize][numClients+1];
    signal sumLocalB[hiddenSize];
    signal oldBscaled[hiddenSize];
    signal diffB[hiddenSize];

    // single-var approach for bias deltas
    component deltaBComp[hiddenSize];
    signal deltaBVal[hiddenSize];

    // mul comps for bias
    component mulCompB[hiddenSize];
    signal tmpMulB[hiddenSize];
    signal checkCB[hiddenSize];

    signal computedGB[hiddenSize];

    //
    // ======== 3) Local model hash checks ========
    // We'll flatten LWp[c], LBp[c] for each c, then compare with ScLH[c].
    //
    component localHashers[numClients];
    component localComparisons[numClients];

    // dimension each local model flatten: totalLocalSize = hiddenSize*inputSize + hiddenSize
    var totalLocalSize = hiddenSize * inputSize + hiddenSize;

    //
    // ======== 4) Confirm final global model (GWp, GBp) & hash (gdigest) ========
    //
    // We'll define signals for diffCw, diffCb, and flatten new model for final hasher
    signal diffCw[hiddenSize][inputSize];
    signal diffCb[hiddenSize];

    // final flatten
    var finalSize = hiddenSize * inputSize + hiddenSize;
    component globalHasher = MiMCArray(finalSize);
    signal finalFlatten[finalSize];

    // ================== LOOPS ==================

    // Weighted average for Weights
    var wIndex = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            // partial sums
            partialSumW[i][j][0] <== 0;
            for (var c = 0; c < numClients; c++) {
                partialSumW[i][j][c+1] <== partialSumW[i][j][c] + LWp[c][i][j];
            }

            sumLocalW[i][j] <== partialSumW[i][j][numClients];
            oldScaledW[i][j] <== GW[i][j] * numClients;
            diffW[i][j] <== sumLocalW[i][j] - oldScaledW[i][j];

            // define a free variable deltaWVal[i][j] via SingleVariable
            deltaWComp[wIndex] = SingleVariable();
            // unify them
            deltaWVal[i][j] <== deltaWComp[wIndex].x;

            // multiply deltaWVal[i][j] * numClients => tmpMulW
            mulCompW[wIndex] = Mul2();
            mulCompW[wIndex].a <== deltaWVal[i][j];
            mulCompW[wIndex].b <== numClients;

            tmpMulW[i][j] <== mulCompW[wIndex].c;
            checkCW[i][j] <== tmpMulW[i][j] - diffW[i][j];
            checkCW[i][j] === 0;

            computedGW[i][j] <== GW[i][j] + deltaWVal[i][j];

            wIndex++;
        }
    }

    // Weighted average for Bias
    for (var i = 0; i < hiddenSize; i++) {
        partialSumB[i][0] <== 0;
        for (var c = 0; c < numClients; c++) {
            partialSumB[i][c+1] <== partialSumB[i][c] + LBp[c][i];
        }

        sumLocalB[i] <== partialSumB[i][numClients];
        oldBscaled[i] <== GB[i] * numClients;
        diffB[i] <== sumLocalB[i] - oldBscaled[i];

        // define free var for deltaBVal[i] 
        deltaBComp[i] = SingleVariable();
        deltaBVal[i] <== deltaBComp[i].x;

        // multiply deltaBVal[i] * numClients => tmpMulB[i]
        mulCompB[i] = Mul2();
        mulCompB[i].a <== deltaBVal[i];
        mulCompB[i].b <== numClients;

        tmpMulB[i] <== mulCompB[i].c;
        checkCB[i] <== tmpMulB[i] - diffB[i];
        checkCB[i] === 0;

        computedGB[i] <== GB[i] + deltaBVal[i];
    }

    // Local model hashes
    for (var c = 0; c < numClients; c++) {
        localHashers[c] = MiMCArray(totalLocalSize);

        var idxLH = 0;
        for (var i = 0; i < hiddenSize; i++) {
            for (var j = 0; j < inputSize; j++) {
                localHashers[c].ins[idxLH] <== LWp[c][i][j];
                idxLH++;
            }
        }
        for (var i = 0; i < hiddenSize; i++) {
            localHashers[c].ins[idxLH] <== LBp[c][i];
            idxLH++;
        }
        localHashers[c].k <== 0;

        localComparisons[c] = HashEquality();
        localComparisons[c].hash1 <== localHashers[c].hash;
        localComparisons[c].hash2 <== ScLH[c];
        localComparisons[c].equal === 0;
    }

    // Confirm computedGW, computedGB == GWp, GBp
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            diffCw[i][j] <== computedGW[i][j] - GWp[i][j];
            diffCw[i][j] === 0;
        }
        diffCb[i] <== computedGB[i] - GBp[i];
        diffCb[i] === 0;
    }

    // Flatten new global for final hash
    var idxFin = 0;
    for (var i = 0; i < hiddenSize; i++) {
        for (var j = 0; j < inputSize; j++) {
            finalFlatten[idxFin] <== GWp[i][j];
            idxFin++;
        }
    }
    for (var i = 0; i < hiddenSize; i++) {
        finalFlatten[idxFin] <== GBp[i];
        idxFin++;
    }

    for (var xF = 0; xF < finalSize; xF++) {
        globalHasher.ins[xF] <== finalFlatten[xF];
    }
    globalHasher.k <== 0;

    component globalHashCompare = HashEquality();
    globalHashCompare.hash1 <== globalHasher.hash;
    globalHashCompare.hash2 <== gdigest;
    globalHashCompare.equal === 0;

    // optional outputs
    signal output validLocalHashes;
    signal output finalModelHashCheck;

    // Summation of localComparisons
    signal sumLocalDiff[numClients+1];
    sumLocalDiff[0] <== 0;
    for (var c = 0; c < numClients; c++) {
        sumLocalDiff[c+1] <== sumLocalDiff[c] + localComparisons[c].equal;
    }
    validLocalHashes <== sumLocalDiff[numClients];

    finalModelHashCheck <== globalHashCompare.equal;
}

// The aggregator main
component main = AggregatorCircuit(4, 5, 10);

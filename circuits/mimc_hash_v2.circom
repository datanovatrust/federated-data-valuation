pragma circom 2.0.0;

/**
 * MiMC Hash Circuits (v2), fully Circom 2.0–compliant.
 * - 2-round MiMC for demonstration
 * - All signals declared outside loops
 */
template MiMCSponge(nInputs) {
    signal input ins[nInputs];
    signal input k;
    signal output hash;

    // minimal 2-round constants
    var constants[2];
    constants[0] = 7919;
    constants[1] = 7927;

    // Pre-declare arrays
    signal state[nInputs + 1];
    signal afterAdd[nInputs];
    signal roundVal[nInputs][3];
    signal tmpVal[nInputs][2];
    signal squaredVal[nInputs][2];
    signal cubedVal[nInputs][2];

    // initialization
    state[0] <== k;

    for (var i = 0; i < nInputs; i++) {
        afterAdd[i] <== state[i] + ins[i];
        roundVal[i][0] <== afterAdd[i];

        for (var r = 0; r < 2; r++) {
            tmpVal[i][r] <== roundVal[i][r] + constants[r];
            squaredVal[i][r] <== tmpVal[i][r] * tmpVal[i][r];
            cubedVal[i][r] <== squaredVal[i][r] * tmpVal[i][r];
            roundVal[i][r+1] <== cubedVal[i][r];
        }
        state[i+1] <== roundVal[i][2];
    }

    hash <== state[nInputs];
}

template MiMCArray(n) {
    signal input ins[n];
    signal input k;
    signal output hash;

    component sponge = MiMCSponge(n);
    for (var i = 0; i < n; i++) {
        sponge.ins[i] <== ins[i];
    }
    sponge.k <== k;

    hash <== sponge.hash;
}

template HashEquality() {
    signal input hash1;
    signal input hash2;
    signal output equal;

    signal diff;
    diff <== hash1 - hash2;
    equal <== diff; // 0 if they match
}

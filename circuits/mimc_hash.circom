// circuits/mimc_hash.circom

pragma circom 2.0.0;

function get_mimc_constants(n) {
    var constants[110];  // Reduced from 220 to 110 rounds
    for (var i = 0; i < n; i++) {
        constants[i] = i + 1;
    }
    return constants;
}

template MiMCSponge(nInputs) {
    signal input ins[nInputs];  
    signal input k;             
    signal output hash;         

    var nRounds = 110;  // Reduced number of rounds
    var constants[110] = get_mimc_constants(nRounds);

    signal currentStateInputs[nInputs+1];
    currentStateInputs[0] <== k;

    signal afterAdd[nInputs];
    signal roundStates[nInputs][nRounds+1];
    signal tVals[nInputs][nRounds];
    signal tSquaredVals[nInputs][nRounds];
    signal tCubedVals[nInputs][nRounds];

    for (var i = 0; i < nInputs; i++) {
        afterAdd[i] <== currentStateInputs[i] + ins[i];
        roundStates[i][0] <== afterAdd[i];

        for (var j = 0; j < nRounds; j++) {
            tVals[i][j] <== roundStates[i][j] + constants[j];
            tSquaredVals[i][j] <== tVals[i][j] * tVals[i][j];
            tCubedVals[i][j] <== tSquaredVals[i][j] * tVals[i][j];
            roundStates[i][j+1] <== tCubedVals[i][j];
        }

        currentStateInputs[i+1] <== roundStates[i][nRounds];
    }

    hash <== currentStateInputs[nInputs];
}

template MiMCTwoInputs() {
    signal input in1;
    signal input in2;
    signal input k;
    signal output hash;

    component mimc = MiMCSponge(2);
    mimc.ins[0] <== in1;
    mimc.ins[1] <== in2;
    mimc.k <== k;
    hash <== mimc.hash;
}

template MiMCArray(n) {
    signal input ins[n];
    signal input k;
    signal output hash;

    component mimc = MiMCSponge(n);
    for (var i = 0; i < n; i++) {
        mimc.ins[i] <== ins[i];
    }
    mimc.k <== k;
    hash <== mimc.hash;
}
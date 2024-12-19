pragma circom 2.0.0;

function get_mimc_constants(n) {
    var constants[2];
    for (var i = 0; i < n; i++) {
        constants[i] = i + 1;
    }
    return constants;
}

template ModAdd() {
    signal input a;
    signal input b;
    signal output out;
    
    out <== a + b;
}

template ModMul() {
    signal input a;
    signal input b;
    signal output out;
    
    out <== a * b;
}

template MiMCSponge(nInputs) {
    signal input ins[nInputs];  
    signal input k;             
    signal output hash;         

    var nRounds = 2;
    var constants[2] = get_mimc_constants(nRounds);

    component adders[nInputs];
    component muls[nInputs][nRounds];

    signal currentStateInputs[nInputs+1];
    currentStateInputs[0] <== k;

    // Declare all signals needed for round computations outside loops:
    signal roundStates[nInputs][nRounds+1];
    signal tvals[nInputs][nRounds];
    signal tSquaredvals[nInputs][nRounds];
    signal tCubedvals[nInputs][nRounds];

    for (var i = 0; i < nInputs; i++) {
        adders[i] = ModAdd();
        adders[i].a <== currentStateInputs[i];
        adders[i].b <== ins[i];

        // Initial round state for this input
        roundStates[i][0] <== adders[i].out;

        for (var j = 0; j < nRounds; j++) {
            tvals[i][j] <== roundStates[i][j] + constants[j];

            muls[i][j] = ModMul();
            muls[i][j].a <== tvals[i][j];
            muls[i][j].b <== tvals[i][j];

            tSquaredvals[i][j] <== muls[i][j].out;
            tCubedvals[i][j] <== tSquaredvals[i][j] * tvals[i][j];
            roundStates[i][j+1] <== tCubedvals[i][j];
        }

        currentStateInputs[i+1] <== roundStates[i][nRounds];
    }

    hash <== currentStateInputs[nInputs];
}

template HashEquality() {
    signal input hash1;
    signal input hash2;
    signal output equal;
    
    signal diff;
    diff <== hash1 - hash2;
    equal <== diff;
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

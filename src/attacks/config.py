# src/attacks/config.py

class RMIAConfig:
    """
    Configuration parameters for the RMIA attack.
    """

    # Threshold for the likelihood ratio test (gamma)
    GAMMA = 1.0

    # Threshold for the membership inference score (beta)
    BETA = 0.5

    # Number of reference models to use
    NUM_REFERENCE_MODELS = 5

    # Number of population data samples (z)
    NUM_Z_SAMPLES = 1000

    # Other hyperparameters can be added here
    # For example, paths to models, data directories, etc.

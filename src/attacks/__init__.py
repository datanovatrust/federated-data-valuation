# src/attacks/__init__.py

# Import classes and functions for easy access
from .rmia_attack import RMIAttack
from .statistical_tests import (
    compute_likelihood_ratio,
    compute_score_mia,
    hypothesis_test
)
from .reference_model_manager import ReferenceModelManager
from .data_sampler import DataSampler
from .evaluation_metrics import (
    compute_tpr_fpr,
    plot_roc_curve,
    compute_auc
)
from .config import RMIAConfig

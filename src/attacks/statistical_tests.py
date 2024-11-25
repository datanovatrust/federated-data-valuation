# src/attacks/statistical_tests.py

import torch

def compute_likelihood_ratio(theta, x, y_x, z, y_z, reference_models):
    """
    Computes the likelihood ratio LR_theta(x, z).
    """
    # Ensure x has batch dimension
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Adds a batch dimension at dim=0

    # Ensure z has batch dimension
    if z.dim() == 3:
        z = z.unsqueeze(0)

    # Compute Pr(x | theta)
    pr_x_given_theta = theta.predict_proba(x)[0, y_x]

    # Compute Pr(x)
    pr_x = compute_average_probability(x, y_x, reference_models)

    # Compute Pr(z | theta)
    pr_z_given_theta = theta.predict_proba(z)[0, y_z]

    # Compute Pr(z)
    pr_z = compute_average_probability(z, y_z, reference_models)

    # Compute the likelihood ratio
    numerator = pr_x_given_theta / pr_x
    denominator = pr_z_given_theta / pr_z
    lr = numerator / denominator
    return lr

def compute_average_probability(x, y, models):
    """
    Computes the average probability Pr(x) over a list of models.
    """
    # Ensure x has batch dimension
    if x.dim() == 3:
        x = x.unsqueeze(0)

    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            prob = model.predict_proba(x)[0, y]
            probs.append(prob)
    avg_prob = sum(probs) / len(probs)
    return avg_prob

def compute_score_mia(likelihood_ratios, gamma):
    """
    Computes the membership inference score.

    Args:
        likelihood_ratios: A list of likelihood ratios.
        gamma: Threshold for the likelihood ratio test.

    Returns:
        score_mia: The membership inference score.
    """
    count = sum(1 for lr in likelihood_ratios if lr >= gamma)
    score_mia = count / len(likelihood_ratios)
    return score_mia

def hypothesis_test(score_mia, beta):
    """
    Performs the final hypothesis test to determine membership.

    Args:
        score_mia: The membership inference score.
        beta: Threshold for the membership inference score.

    Returns:
        is_member: True if x is predicted to be a member, False otherwise.
    """
    return score_mia >= beta

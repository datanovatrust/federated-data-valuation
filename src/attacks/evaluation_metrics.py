# src/attacks/evaluation_metrics.py

from sklearn.metrics import roc_curve, auc

def compute_tpr_fpr(true_labels, pred_scores):
    """
    Computes true positive rates and false positive rates.

    Args:
        true_labels: Ground truth labels (1 for members, 0 for non-members).
        pred_scores: Predicted membership inference scores.

    Returns:
        fpr: False positive rates.
        tpr: True positive rates.
    """
    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    return fpr, tpr

def compute_auc(fpr, tpr):
    """
    Computes the Area Under the ROC Curve.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.

    Returns:
        auc_score: The computed AUC score.
    """
    auc_score = auc(fpr, tpr)
    return auc_score

def plot_roc_curve(fpr, tpr, title='ROC Curve', save_path=None):
    """
    Plots the ROC curve.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        title: Title of the plot.
        save_path: Path to save the plot image file. If None, the plot is displayed.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

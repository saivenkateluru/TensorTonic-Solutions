import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    #y_pred = max(prob_pred)
    y_true = np.asarray(y_true, dtype = int)
    prob_pred = np.asarray(y_pred, dtype = float)
    avg_ce_loss = -1 * np.mean(np.log(prob_pred[np.arange(len(y_true)),y_true]))
    return avg_ce_loss
import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    #1/1+exp(-x)
    x = np.asarray(x, dtype=float)
    ex_p = np.exp(-1*x)
    return 1/(1+ex_p)
    
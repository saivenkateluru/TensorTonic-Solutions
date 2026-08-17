import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    #1/1+exp(-x)
    x = np.array(x)
    ex_p = np.exp(-1*x)
    return 1/(1+ex_p)
    
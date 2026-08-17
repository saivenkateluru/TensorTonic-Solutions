import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    #relu(x) = max(0,x)
    x = np.asarray(x , dtype=float)
    zers = np.zeros(x.shape)
    return np.max((zers,x),axis = 0)
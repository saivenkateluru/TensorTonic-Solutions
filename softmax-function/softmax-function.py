import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x,dtype=float)
    m = np.max(x,axis=-1,keepdims=True)
    shifted = x-m
    softmx = np.exp(shifted) / (np.sum(np.exp(shifted),axis=-1,keepdims=True))
    return softmx
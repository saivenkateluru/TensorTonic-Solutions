def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    x = np.array(X)
    w = np.array(W)
    b_ = np.array(b)
    return (np.einsum("ij,jk->ik", x, w) + b_).tolist()
    
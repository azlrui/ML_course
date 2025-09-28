import numpy as np
from .costs import compute_loss

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************
    tol = 1e-5
    e = y - tx @ w
    s = np.sign(e)
    nondiff = np.flatnonzero(np.abs(e) < tol)
    
    if nondiff.size > 0:
        # If there are non-differentiable points, we can set their sign to 0
        s[nondiff] = 0  # to avoid numerical issues with sign function

    subgradient_w = - np.sign(e) @ tx / (len(y) * 0.5)

    return subgradient_w, nondiff

def subgradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    """The SubGradient Descent (SubGD) algorithm for MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SubGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SubGD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute subgradient and loss
        # ***************************************************
        subgradient, ind = compute_subgradient_mae(y, tx, w)
        loss = compute_loss(y, tx, w, method="MAE")
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by subgradient
        # ***************************************************
        w = w - gamma * subgradient

        ws.append(w)
        losses.append(loss)
        print(
            "SubGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}, non-diff point = {ind}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], ind=ind.size
            )
        )

    return losses, ws
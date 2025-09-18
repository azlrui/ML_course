# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y:np.array, tx:np.array, w:np.array, method="MSE") -> float:
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    e = y - tx.dot(w)
    if method == "MSE":
        return 1 / (2 * len(y)) * np.sum(e ** 2)
    elif method == "MAE":
        return 1 / len(y) * np.sum(np.abs(e))
    else:
        raise ValueError("Invalid method. Use 'MSE' or 'MAE'.")

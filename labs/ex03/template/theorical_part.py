import sys, os

repo_root = r"s:\Documents\.Code\GitHub\ML_course"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import matplotlib.pyplot as plt
from helpers import *
import labs.ex02.template.costs as cs

def weighted_mse(y, X, w, eps=1e-8):
    """
    y: (N,), X: (N,D), w: (D,)
    returns scalar loss = (1/(2N)) * sum_i ((Xw - y)^2 / (y^2 + eps))
    """
    r = X @ w - y                          # (N,)
    denom = y**2 + eps                     # (N,)
    return 0.5 * np.mean((r**2) / denom)

def weighted_least_squares(y, X, eps=1e-8):
    W = 1.0 / (y**2 + eps)                 # (N,)
    # compute A = X^T W X and b = X^T W y without forming dense diag(W)
    A = X.T @ (W[:, None] * X)             # (D,D)
    b = X.T @ (W * y)                      # (D,)
    w = np.linalg.solve(A, b)
    return w

def run_test_with_data():
    x, y = load_data()
    N = len(y)

    x = np.column_stack((np.ones(N), x))

    w = weighted_least_squares(y, x)

    loss = weighted_mse(y, x, w)

    print(loss)

if __name__ == "__main__":

    run_test_with_data()




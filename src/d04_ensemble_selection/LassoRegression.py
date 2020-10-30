# adapted from https://xavierbourretsicotte.github.io/lasso_implementation.html

import numpy as np


def soft_threshold(rho, alpha):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - alpha:
        return (rho + alpha)
    elif rho > alpha:
        return (rho - alpha)
    else:
        return 0


def coordinate_descent_lasso(theta, X, y, alpha=.01, num_iters=100, intercept=False):
    '''Coordinate gradient descent for lasso regression - for normalized data.
    The intercept parameter allows to specify whether or not we regularize theta_0'''

    # Initialisation of useful values
    m, n = X.shape
    X = X / (np.linalg.norm(X, axis=0))  # normalizing X in case it was not done before

    zeros = np.zeros((256,1))

    # Looping until max number of iterations
    for i in range(num_iters):
        print(f"Alpha: {alpha}, Iteration: {i}")
        # Looping through each coordinate
        for j in range(n):

            # Vectorized implementation
            X_j = X[:, j].reshape(-1, 1)
            y_pred = X @ theta

            # normal
            # rho = X_j.T @ (y - y_pred + theta[j] * X_j)
            # only penalize high values
            dy = y - y_pred
            #dy[dy > 0] = 0
            dy = np.where(dy > 0, 0.95*dy, dy)
            rho = X_j.T @ (dy + theta[j] * X_j)

            # enforce nonnegativity of coeffs
            if rho < 0:
                rho = 0

            # Checking intercept parameter
            if intercept == True:
                if j == 0:
                    theta[j] = rho
                else:
                    theta[j] = soft_threshold(rho, alpha)

            if intercept == False:
                theta[j] = soft_threshold(rho, alpha)

    return theta.flatten(), y_pred

import numpy as np
from numpy.linalg import eig, cholesky


def make_cov_posdef(cov):
    if not is_pos_def(cov):
        print('Warning: Covariance matrix is not positive definite')
        print('The covariance matrix will be regularized')
        jitter = 1e-15 * np.eye(cov.shape[0])
        w, v = eig(cov+jitter)
        w = np.real(np.abs(w))
        v = np.real(v)
        cov = v @ np.diag(np.abs(w)) @ v.T
        cov = np.tril(cov) + np.triu(cov.T, 1)
        if not is_pos_def(cov):
            print('Warning: regularization failed')
            print('The covariance matrix will be diagonalized')
            jitter = 1e-15
            cov = np.diag(np.diag(cov)+jitter)
    return cov


def is_pos_def(A):
    try:
        cholesky(A)
        return True
    except np.linalg.linalg.LinAlgError as err:
        return False

import numpy as np

def calc_sh(ub):
    """
    Calculates the estimator ŝ = (A.T*A).I*A.T*ub1
    """
    ub = ub.T # Lines are timestamps.
    # Apply window to data matrix.
    for i in range(1-Lh, 1+Lhnt):
        A[i,:] = ub[i,:]

    AT = A.T
    np.linalg.solve(AT*A, AT*ub1)

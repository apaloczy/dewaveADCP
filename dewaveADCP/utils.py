import numpy as np
from numpy.linalg import lstsq, solve


def lstsqfit(d, r, n=1):
    """
    USAGE
    -----
    Gm = lstsq(d, r, n=1)

    Fit an 'n'-th order polynomial to the input data vector by solving an overdetermined least-squares problem of degree 'n' (default 1).
    """
    d = np.matrix(np.array(d)).T
    nd = d.size
    if n>nd:
        Error("Polynomial degree ('%d') is larger than number of points ('%d')"%(n,nd))

    # Set up data kernel matrix G and solve.
    G = np.matrix(np.array([], ndmin=2))

    for pw in range(n, 0, -1):
        mm = np.matrix(r**pw).T
        if pw==n:
            G = mm
        else:
            G = np.hstack((G, mm))

    col1s = np.matrix(np.ones((nd, 1)))
    G = np.hstack((G, col1s))

    GT = G.T
    m = solve(GT*G, GT*d) # m = (GT*G).I*GT*d.T

    return G*m


d2r = np.pi/180
def sind(ang):
    return np.sin(ang*d2r)


def cosd(ang):
    return np.cos(ang*d2r)
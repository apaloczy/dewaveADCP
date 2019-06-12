import numpy as np
from xarray import DataArray


def bvar4AF(b1, b2, b3, b4, t, theta, sep=6, Lw=128, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1var, b2var, b3var, b4var = bvar4AF(b1, b2, b3, b4, t, theta, sep=6, Lw=128, max_badfrac=0.3, verbose=False)

    Acknowledgement: Part of this function was based on
    original MATLAB code kindly provided by Johanna Rosman.
    """
    nz, nt = b1.shape
    nb = 4
    b1, b2, b3, b4 = b1[np.newaxis,...], b2[np.newaxis,...], b3[np.newaxis,...], b4[np.newaxis,...]
    B = np.vstack((b1, b2, b3, b4))

    Bvar = np.empty((nb, nz, nt))*np.nan

    for m in range(nb):
        ub = B[m,...]
        ub = ub.T # Lines are timestamps.
        for k in range(nz-sep):
            ub1 = ub[:,k]
            ub2 = ub[:,k+sep]

            fbad0 = np.isnan(ub2)
            ub1 = DataArray(ub1, coords=dict(t=t), dims='t').interpolate_na(dim='t').values
            ub2 = DataArray(ub2, coords=dict(t=t), dims='t').interpolate_na(dim='t').values

            # Drop remaining NaNs after interpolating.
            fbad = np.isnan(ub2)
            if verbose:
                print("Interpolated over %d NaNs."%(fbad0.sum()-fbad.sum()))
            ub1 = ub1[~fbad]
            ub2 = ub2[~fbad]
            ntt = ub2.size
            if (fbad0.sum()/nt)>max_badfrac:
                if verbose:
                    print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                continue
            else:
                if verbose:
                    print("Removed %d NaNs."%fbad.sum())

            # Make windowed data matrix A from velocities at bin 2.
            A = np.matrix(np.empty((ntt-Lw, Lw)))*np.nan
            for i in range(0, ntt-Lw):
                A[i,:] = ub2[i:i+Lw]

            # Calculate the weights estimator ŝ = (A.T*A).I*A.T*ub1 and
            # use ŝ to calculate ub1h, the predicted velocities at the
            # *lower* bin, based on velocities at the *upper* bin.
            AT = A.T
            try:
                s = np.linalg.solve(AT*A, AT*np.matrix(ub1[:-Lw]).T)
            except np.linalg.LinAlgError:
                if verbose:
                    print("Singular matrix, skipping bin #%d."%k)
                continue
            ub1h = np.array(A*s).squeeze()
            ub1d = ub1[:-Lw] - ub1h
            Bvar[m, k, :ntt-Lw] = ub1[:-Lw]*ub1d

    # Variances corrected for the wave bias.
    b1var, b2var = Bvar[0, ...], Bvar[1, ...]
    b3var, b4var = Bvar[2, ...], Bvar[3, ...]

    return b1var, b2var, b3var, b4var


def bvar5AF(b1, b2, b3, b4, b5, t, theta, sep=6, Lw=128, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1var, b2var, b3var, b4var, b5var = bvar5AF(b1, b2, b3, b4, b5, t, theta, sep=6, Lw=128, max_badfrac=0.3, verbose=False)

    Acknowledgement: Part of this function was based on
    original MATLAB code kindly provided by Johanna Rosman.
    """
    nz, nt = b1.shape
    nb = 5
    b1, b2, b3, b4, b5 = b1[np.newaxis,...], b2[np.newaxis,...], b3[np.newaxis,...], b4[np.newaxis,...], b5[np.newaxis,...]
    B = np.vstack((b1, b2, b3, b4, b5))

    Bvar = np.empty((nb, nz, nt))*np.nan

    for m in range(nb):
        ub = B[m,...]
        ub = ub.T # Lines are timestamps.
        for k in range(nz-sep):
            ub1 = ub[:,k]
            ub2 = ub[:,k+sep]

            fbad0 = np.isnan(ub2)
            ub1 = DataArray(ub1, coords=dict(t=t), dims='t').interpolate_na(dim='t').values
            ub2 = DataArray(ub2, coords=dict(t=t), dims='t').interpolate_na(dim='t').values

            # Drop remaining NaNs after interpolating.
            fbad = np.isnan(ub2)
            if verbose:
                print("Interpolated over %d NaNs."%(fbad0.sum()-fbad.sum()))
            ub1 = ub1[~fbad]
            ub2 = ub2[~fbad]
            ntt = ub2.size
            if (fbad0.sum()/nt)>max_badfrac:
                if verbose:
                    print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                continue
            else:
                if verbose:
                    print("Removed %d NaNs."%fbad.sum())

            # Make windowed data matrix A from velocities at bin 2.
            A = np.matrix(np.empty((ntt-Lw, Lw)))*np.nan
            for i in range(0, ntt-Lw):
                A[i,:] = ub2[i:i+Lw]

            # Calculate the weights estimator ŝ = (A.T*A).I*A.T*ub1 and
            # use ŝ to calculate ub1h, the predicted velocities at the
            # *lower* bin, based on velocities at the *upper* bin.
            AT = A.T
            try:
                s = np.linalg.solve(AT*A, AT*np.matrix(ub1[:-Lw]).T)
            except np.linalg.LinAlgError:
                if verbose:
                    print("Singular matrix, skipping bin #%d."%k)
                continue
            ub1h = np.array(A*s).squeeze()
            ub1d = ub1[:-Lw] - ub1h
            Bvar[m, k, :ntt-Lw] = ub1[:-Lw]*ub1d

    # Variances corrected for the wave bias.
    b1var, b2var = Bvar[0, ...], Bvar[1, ...]
    b3var, b4var, b5var = Bvar[2, ...], Bvar[3, ...], Bvar[4, ...]

    return b1var, b2var, b3var, b4var, b5var

import numpy as np
from xarray import DataArray


def interpub1ub2(ub1, ub2, t, max_badfrac, k, verbose=False):
    fbad01 = np.isnan(ub1)
    fbad02 = np.isnan(ub2)
    ub1 = DataArray(ub1, coords=dict(t=t), dims='t').interpolate_na(dim='t').values
    ub2 = DataArray(ub2, coords=dict(t=t), dims='t').interpolate_na(dim='t').values

    # Clip remaining NaNs after interpolating.
    fg = ~np.logical_or(np.isnan(ub1), np.isnan(ub2))
    ub1 = ub1[fg]
    ub2 = ub2[fg]
    nt = t.size
    if (np.maximum(fbad01.sum()/nt, fbad02.sum()/nt))>max_badfrac: # If either bin had too many NaNs, interpolated too much, skip.
        if verbose:
            print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
        return None, None
    else:
        if verbose:
            nfbad = np.sum(~fg)
            print("Interpolated over %d/%d NaNs in the bottom/top bin."%(fbad01.sum()-nfbad, fbad02.sum()-nfbad))
            print("After interpolation, removed %d additional NaNs in both bins."%nfbad)

    return ub1, ub2

def bvar4AF(b1, b2, b3, b4, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1var, b2var, b3var, b4var = bvar4AF(b1, b2, b3, b4, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False)

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

            if interpolate:
                ub1, ub2 = interpub1ub2(ub1, ub2, t, max_badfrac, k, verbose=verbose)
                if ub1 is None:
                    continue
            else: # Take only timestamps where both levels are non-NaN.
                fub1ub2 = np.logical_and(~np.isnan(ub1), ~np.isnan(ub2))
                ub1 = ub1[fub1ub2]
                ub2 = ub2[fub1ub2]
                percg = fub1ub2.sum()/nt
                if percg<(1-max_badfrac):
                    if verbose:
                        print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                    continue
                if verbose:
                    print("Using only non-NaNs (%.1f%% data points, bin #%d)"%(100*percg,k+1))

            # Make windowed data matrix A from velocities at bin 2.
            ntt = ub2.size
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


def bvar5AF(b1, b2, b3, b4, b5, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1var, b2var, b3var, b4var, b5var = bvar5AF(b1, b2, b3, b4, b5, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False)

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

            if interpolate:
                ub1, ub2 = interpub1ub2(ub1, ub2, t, max_badfrac, k, verbose=verbose)
                if ub1 is None:
                    continue
            else: # Take only timestamps where both levels are non-NaN.
                fub1ub2 = np.logical_and(~np.isnan(ub1), ~np.isnan(ub2))
                ub1 = ub1[fub1ub2]
                ub2 = ub2[fub1ub2]
                percg = fub1ub2.sum()/nt
                if percg<(1-max_badfrac):
                    if verbose:
                        print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                    continue
                if verbose:
                    print("Using only non-NaNs (%.1f%% data points, bin #%d)"%(100*percg,k+1))

            # Make windowed data matrix A from velocities at bin 2.
            ntt = ub2.size
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


def bvel4AF(b1, b2, b3, b4, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1, b2, b3, b4, b5 = bvel4AF(b1, b2, b3, b4, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False)

    Acknowledgement: Part of this function was based on
    original MATLAB code kindly provided by Johanna Rosman.
    """
    nz, nt = b1.shape
    nb = 4
    b1, b2, b3, b4 = b1[np.newaxis,...], b2[np.newaxis,...], b3[np.newaxis,...], b4[np.newaxis,...]
    B = np.vstack((b1, b2, b3, b4))

    Bvel = np.empty((nb, nz, nt))*np.nan

    for m in range(nb):
        ub = B[m,...]
        ub = ub.T # Lines are timestamps.
        for k in range(nz-sep):
            ub1 = ub[:,k]
            ub2 = ub[:,k+sep]

            if interpolate:
                ub1, ub2 = interpub1ub2(ub1, ub2, t, max_badfrac, k, verbose=verbose)
                if ub1 is None:
                    continue
            else: # Take only timestamps where both levels are non-NaN.
                fub1ub2 = np.logical_and(~np.isnan(ub1), ~np.isnan(ub2))
                ub1 = ub1[fub1ub2]
                ub2 = ub2[fub1ub2]
                percg = fub1ub2.sum()/nt
                if percg<(1-max_badfrac):
                    if verbose:
                        print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                    continue
                if verbose:
                    print("Using only non-NaNs (%.1f%% data points, bin #%d)"%(100*percg,k+1))

            # Make windowed data matrix A from velocities at bin 2.
            ntt = ub2.size
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
            Bvel[m, k, :ntt-Lw] = ub1d

    # Variances corrected for the wave bias.
    b1vel, b2vel = Bvel[0, ...], Bvel[1, ...]
    b3vel, b4vel = Bvel[2, ...], Bvel[3, ...]

    return b1vel, b2vel, b3vel, b4vel


def bvel5AF(b1, b2, b3, b4, b5, t, sep=6, Lw=128, interpolate=False, max_badfrac=0.3, verbose=False):
    """
    USAGE
    -----
    b1, b2, b3, b4, b5 = bvel5AF(b1, b2, b3, b4, b5, t, sep=6, Lw=128, max_badfrac=0.3, verbose=False)

    Acknowledgement: Part of this function was based on
    original MATLAB code kindly provided by Johanna Rosman.
    """
    nz, nt = b1.shape
    nb = 5
    b1, b2, b3, b4, b5 = b1[np.newaxis,...], b2[np.newaxis,...], b3[np.newaxis,...], b4[np.newaxis,...], b5[np.newaxis,...]
    B = np.vstack((b1, b2, b3, b4, b5))

    Bvel = np.empty((nb, nz, nt))*np.nan

    for m in range(nb):
        ub = B[m,...]
        ub = ub.T # Lines are timestamps.
        for k in range(nz-sep):
            ub1 = ub[:,k]
            ub2 = ub[:,k+sep]

            if interpolate:
                ub1, ub2 = interpub1ub2(ub1, ub2, t, max_badfrac, k, verbose=verbose)
                if ub1 is None:
                    continue
            else: # Take only timestamps where both levels are non-NaN.
                fub1ub2 = np.logical_and(~np.isnan(ub1), ~np.isnan(ub2))
                ub1 = ub1[fub1ub2]
                ub2 = ub2[fub1ub2]
                percg = fub1ub2.sum()/nt
                if percg<(1-max_badfrac):
                    if verbose:
                        print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
                    continue
                if verbose:
                    print("Using only non-NaNs (%.1f%% data points, bin #%d)"%(100*percg,k+1))

            # Make windowed data matrix A from velocities at bin 2.
            ntt = ub2.size
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
            Bvel[m, k, :ntt-Lw] = ub1d

    # Variances corrected for the wave bias.
    b1vel, b2vel = Bvel[0, ...], Bvel[1, ...]
    b3vel, b4vel, b5vel = Bvel[2, ...], Bvel[3, ...], Bvel[4, ...]

    return b1vel, b2vel, b3vel, b4vel, b5vel

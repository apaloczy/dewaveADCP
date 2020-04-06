import numpy as np
from xarray import DataArray


def bvarAF(b, sep=6, Lw=8, max_badfrac=0.5, mindet=1e-4, verbose=False):
    """
    USAGE
    -----
    bvar_dewaved = bvarAF(bvel, sep=6, Lw=8, max_badfrac=0.5, verbose=False)
    """
    nz, nt = b.shape
    bvardw = np.empty(nz)*np.nan
    b = b.T # Each row is a timestamp.
    for k in range(nz-sep):
        ub1 = b[:,k]
        ub2 = b[:,k+sep]
        # Take only timestamps where both levels are non-NaN.
        fub1ub2 = np.logical_and(np.isfinite(ub1), np.isfinite(ub2))
        percg = fub1ub2.sum()/nt
        if percg<(1-max_badfrac):
            if verbose:
                print("More than ", 100*max_badfrac,"%% NaNs, skipping bin #%d."%k)
            continue
        if verbose:
            print("Using only non-NaNs (%.1f%% data points, bin #%d)"%(100*percg,k+1))

        ub1 = ub1[fub1ub2]
        ub2 = ub2[fub1ub2]

        # Remove time-means.
        ub1 = ub1 - ub1.mean()
        ub2 = ub2 - ub2.mean()

        # Make windowed data matrix A from velocities at bin 2 (upper).
        ntt = ub2.size
        A = np.matrix(np.empty((ntt-Lw, Lw)))*np.nan
        for i in range(0, ntt-Lw):
            A[i,:] = ub2[i:i+Lw]

        # Calculate the weights estimator ŝ = (A.T*A).I*A.T*ub1 and
        # use ŝ to calculate ub1h, the predicted velocities at the
        # *lower* bin, based on velocities at the *upper* bin.
        AT = A.T
        ATA = AT*A
        detATA = np.abs(np.linalg.det(ATA))
        if detATA<mindet:
            if verbose:
                print("AT*A determinant too low (%e), skipping bin #%d."%(detATA, k))
            continue
        else:
            try:
                s = np.linalg.solve(ATA, AT*np.matrix(ub1[:-Lw]).T)
            except np.linalg.LinAlgError:
                if verbose:
                    print("Singular matrix, skipping bin #%d."%(k+1))
                continue
        ub1h = np.array(A*s).squeeze() # Vertically coherent velocity, assumed to be due to waves only.
        ub1d = ub1[:-Lw] - ub1h        # De-waved velocity.

        bvardw[k] = np.mean((ub1d - ub1d.mean())**2)

    return bvardw


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

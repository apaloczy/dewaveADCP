import numpy as np
from scipy import polyfit, polyval
from scipy.optimize import curve_fit
from utils import sind, cosd, lstsqfit


def fexp(x, a, b, c):
    return a + b*np.exp(c*x)


def dewave_verticaldetrend(b, r, theta, ptch, roll, dpoly='exp', params_guess=(-7e-2, 1e-1, -1), detrend_time=True, scipy_detrend=True):
    """
    Fits a line (or a higher-degree polynomial) to the
    *vertical* component of the along-beam velocities.
    """
    b, r = map(np.array, (b, r))
    nt = roll.size
    z = r*cosd(theta)
    if detrend_time:
        b -= np.nanmean(b, axis=1)[..., np.newaxis]

    # *Vertical* component of along-beam velocity, positive down.
    b2z = cosd(theta)*cosd(ptch)*cosd(roll)
    bz = b*b2z

    if dpoly=='exp': # Nonlinear exponential fit.
        for i in range(nt):
            fgud = ~np.isnan(bz[:, i])
            if fgud.sum()==0:
                continue
            else:
                bzi = bz[fgud, i]
                try:
                    params, _ = curve_fit(fexp, z[fgud], bzi, p0=params_guess, maxfev=100000)
                except RuntimeError:
                    Warning('curve_fit() fail. Skipping this profile.')
                    continue
                a, b, c = params
                bzfit = fexp(z, a, b, c)
                bzfit[~fgud] = np.nan
                bz[:, i] = bz[:, i] - bzfit
    else: # Regular least-squares polynomial fit.
        ztrend_bz = b*np.nan
        if scipy_detrend:
            for i in range(nt):
                fgud = ~np.isnan(bz[:, i])
                if fgud.sum()==0:
                    ztrend_bz[:, i] = np.nan
                    continue
                else:
                    bzcff = polyfit(z[fgud], bz[fgud, i], dpoly)
                    ztrend_bz[:, i] = polyval(bzcff, z)
        else:
            for i in range(nt):
                ztrend_bz[:, i] = lstsqfit(bz[:, i], z, n=dpoly).squeeze()

        bz -= ztrend_bz # Remove vertical trend.

    return bz/b2z       # Back to along-beam coordinates (r).

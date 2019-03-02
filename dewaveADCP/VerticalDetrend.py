import numpy as np
from scipy import polyfit, polyval
# from .utils import sind, cosd, lstsqfit
from utils import sind, cosd, lstsqfit

def dewave_verticaldetrend(b, r, theta, ptch, roll, dpoly=1, detrend_time=True, scipy_detrend=True):
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

    ztrend_bz = b*np.nan
    if scipy_detrend:
        for i in range(nt):
            fgud = ~np.isnan(bz[:, i])
            zz, bzz = z[fgud], bz[fgud, i]
            bzcff = polyfit(zz, bzz, dpoly)
            ztrend_bz[:, i] = polyval(bzcff, z)
    else:
        for i in range(nt):
            ztrend_bz[:, i] = lstsqfit(bz[:, i], z, n=dpoly).squeeze()

    bz -= ztrend_bz # Remove vertical trend.
    b = bz/b2z      # Back to along-beam coordinates (r).

    return b

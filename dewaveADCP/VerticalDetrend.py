import numpy as np
from scipy import polyfit, polyval
from .utils import sind, cosd, lstsqfit

def dewave_verticaldetrend(b, r, theta, ptch, roll, dpoly=1, detrend_time=True, use_scipy=True):
    """
    Fits a line (or a higher-degree polynomial) to the
    *vertical* component of the along-beam velocities.
    """
    b, r = map(np.array, (b, r))
    z = r*cosd(theta)
    if detrend_time:
        b -= np.nanmean(b, axis=1)

    # *Vertical* component of along-beam velocity, positive down.
    b2z = cosd(theta)*cosd(ptch)*cosd(roll)
    bz = b*b2z

    if use_scipy:
        bzcff = polyfit(z, bz, dpoly)
        ztrend_bz = polyval(bzcff, z)
    else:
        ztrend_bz = lstsqfit(bz, z, n=dpoly)

    bz -= ztrend_bz # Remove trend.
    b = bz/b2z # Back to along-beam coordinates (r).

    return b

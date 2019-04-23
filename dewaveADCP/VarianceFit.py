import numpy as np
from scipy.optimize import curve_fit
from .utils import sind, cosd


# Whipple et al. (2006) assume waves propagate in beams 1-2's direction:
# bwvar = c1*(cosh(c2*(z + h)) - cosd(2*theta) )
# Rosman et al. (2008) allow for an angle alpha
# between ADCP's beams 1-2 axis and wavenumber's direction:
# c1*((A + B)*cosh(c2*(z + h)) + (A - B)), with A, B defined below.
def sgwvar_func(h, theta, alpha):
    def fsgwvar(z, c1, c2):
        A = (cosd(alpha)*sind(theta))**2
        B = cosd(theta)**2
        return c1*((A + B)*np.cosh(c2*(z + h)) + (A - B))

    return fsgwvar


# c1 = ( (H*om)/(4*np.sinh(k*h)) )**2
# c2 = 2*k
def varfitw(z, bvar, fsgwvar, c1c2guess=(0.2, 0.4)):
    """
    Fit a theoretical variance curve to observed along-beam
    profile of variance 'bvar(z)'. 'fsgwvar' is an instance of
    sgwvar_func(h, theta, alpha), where 'h', 'theta' and 'alpha'
    are the local depth, the Janus beams' angle to the ADCP's
    vertical axis and 'alpha' is the angle between the ADCP's
    x-axis (beams 1-2 for TRDI, 1-3 for Nortek) and the surface
    gravity waves' wavenumber vector.
    """
    fnan = np.isnan(bvar)
    fnnan = ~fnan
    if fnan.sum()>4:
        Warning('Too many NaNs on beam variance profile. Skipping this profile')
        return bvar

    print("Using guess c1, c2 = %.f, %.f for exp fit."%c1c2guess)
    z0, bvar0 = z[fnnan], bvar[fnnan]
    c1c2, _ = curve_fit(fsgwvar, z0, bvar0, p0=c1c2guess, maxfev=10000)

    c1, c2 = c1c2
    bvarfit = fsgwvar(z, c1, c2)
    bvarfit[fnan] = np.nan

    return bvarfit


def calc_beta(bvarfit, dr):
    """
    Calculates Whipple et al.'s (2006) 'beta' parameter based on the
    fit of the wave along-beam variance profile and a specified separation.
    """
    assert dr>0, "Bin separation must be at least 1 bin."
    nbins = bvarfit.size
    zidxbeta = np.arange(dr, nbins-dr, 1)
    beta = []
    for iz in zidxbeta:
        beta.append(np.sqrt(bvarfit[iz-dr]/bvarfit[iz+dr]))

    return np.array(beta), zidxbeta


def dewave_variancefit(b, r, h, hdng, ptch, roll, scipy_detrend=True):
    raise NotImplementedError

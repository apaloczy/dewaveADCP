import numpy as np


def dewave_structurefuntion():
    raise NotImplementedError


def calcDLL(vp, rx=None, remove_shear=True, how_odd='mean_sqseps'):
    """
    USAGE
    -----
    DLL(x, rx) = calcD(vp, rx=None, remove_shear=True, how_odd='mean_sqseps')

    Input
    -----
    vp: Turbulent velocity array from a High-frequency (~1 Hz or faster) ADCP in pulse-to-pulse coherent mode with background shear removed.  Array size [nbeams x nbins x ntimestamps x nensavg]
    where nbeams is number of beams, nbins is number of
    bins per beam, ntimestamps is the number of ensembles per observational
    period (epoch) (i.e. the averging period), and nensavg is the
    number of observational periods.

    rx: [optional] Vector of separations (in bins) over which the velocity
    difference is calculated in numbers of bins.  If omitted,
    differences for all possible values is used

    option: [optional] Determines which averaging approach for separations
    equivalent to an odd number of bins.  option = 1 is the mean of the
    squared difference [default, used if option is not specified]
    option = 2 is square of the mean of the absolute difference option
    = 3 is the square of the mean of the difference

    OUTPUT
    ------
    DLL: Secind-order structure function array with dimensions [nbeams x nbins x nensavg x nr],
    where 'nr' is the length of rx vector, i.e., the number of separation distances at which the
    structure function D is resolved.

    This is a direct translation of the Matlab function 'ADCP_SF_calc_D.m' by Brian Scannell, University of Bangor, which was in turn based on original code developed by Phil Wiles, described in the following references:

    *Wiles, PJ et al (2005) A Novel Technique for Measuring Mixing Using Acoustic Doppler Current
    Profilers (ADCPs), Proceedings of the IEEE/OESEighth Working Conference on Current Measurement
    Technology, doi:10.1109/CCM.2005.1506324.

    *Wiles, PJ et al (2006), A novel technique for measuring the rate of turbulent dissipation in
    the marine environment, Geophys. Res. Lett., 33(21), 1?5, doi:10.1029/2006GL027050.
    """
    if rx is not None:
        rx = np.int32(rx)
        badrx = rx.max()>nbins-1 or rx.min()<2 or np.any(np.ceil(rx)!=np.floor(rx))
            error('*** ERROR *** rx must be vector of integer values with maximum range 2 to nbins-2')
    else:
        rx = np.arange(1, nbins-1, 1)
    nr = rx.size

    nbeams, nbins, ntimestamps, nensavg = vp.shape
    DLL = np.empty((nbeams, nbins, ntimestamps, nr))*np.nan

    # calculate square of velocity differences.
    for n in range(nr):
        rn = rx[n]
        if not r%2: # This separation is an *even* number of bins.
            rnm = rn//2
            ix = np.arange(rnm, nbins - rnm + 1, 1)
            DLL[:, ix, :, :, n] = (vp[:, ix+rnm, :, :] - vp[:, ix-rnm, :, :])**2
        else:       # This separation is an *odd* number of bins.
            rnlm = np.floor(rn/2)
            rnhm = np.ceil(rn/2)
            ix = np.arange(rnhm, nbins - rnhm + 1, 1)
            ixl = ix - rnhm
            ixh = ix - rnlm
            vnl = vp[:, ixl, :, :]
            vnh = vp[:, ixh, :, :]
            vnll = vp[:, ixl+rn, :, :]
            vnhh = vp[:, ixh+rn, :, :]
            if how_odd=='mean_sqseps': # Mean of squared differences.
                DLL[:, ix, :, :, n] = 0.5*((vnll - vnl)**2 + (vnhh - vnh)**2)
            elif how_odd=='asdas':
                DLL[:, ix, :, :, n] = 0.5*(np.abs(vnll - vnl) + np.abs(vnhh - vnh))
            elif how_odd=='mean_seps'
                DLL[:, ix, :, :, n] = (0.5*(vnll - vnl + vnhh - vnh))**2
            else:
                raise NameError("Invalid treatment option '%s' for odd separations."%how_odd)

        # Average the square velocities across timestamps within an ensemble average to give the
        # final structure function array.
        DLL = np.squeeze(np.nanmean(D, axis=2))

        return DLL


def calcepsilon(DLL, rx, C2=2.0, method='modified'):
    """
    USAGE
    -----
    epsilon = calcepsilon(DLL, method='scannell_etal2017')

    C2: Turbulent constant in the Kolmogorov scaling

    Calculate Turbulent Kinetic energy (TKE) rate of dissipation (epsilon) from the second-order
    turbulence structure function (DLL) derived from along-beam ADCP velocity measurements.
    """
    r23 = rx**(2/3) # [m^{2/3}].
    # Fit line (or cubic polynomial if method=='modified') to the measured structure function DLL(x,r**(2/3)).
    # In the cubic-polynomial case (Scannell et al., 2017),
    # DLL(x,r) has a quadratic dependence on 'r', since [r**(2/3)]**3 = r**2.
    if method=='standard': # Fit a line to r**(2/3).
    epsilon = (A1/C2)**(3/2)

    return None

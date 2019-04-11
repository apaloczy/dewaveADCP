import numpy as np
from scipy import polyfit


def dewave_structurefuntion():
    raise NotImplementedError


def calcvp(v, normalize=True):
    """
    USAGE
    -----
    vp = calcvp(v, normalize=True)

    Calculate turbulent velocity vp (v-primed) from full along-beam velocities.
    "v" should have [nbeams x nbins x nens x nepochs] shape.

    ***This is a direct translation of the Matlab function 'ADCP_SF_calc_vp.m'
    by Brian Scannell, University of Bangor.

    Translated by André Palóczy, Jan/2019.
    apaloczy@ucsd.edu
    % function vp = ADCP_SF_calc_vp(v,option)
    %
    % Function to calculate the turbulent velocity component v' from an array of
    % velocity measurement produced by an HF ADCP operating in pulse-pulse
    % coherent mode.  Calculated by deducting the mean background flow from the
    % individual observations.
    %
    % Option to normalise the velocities by deducting the ensemble median from
    % all bins in order to remove the static difference between ensembles prior
    % to calculating the background shear.
    %
    % Inputs
    %   v:  velocity array with dimensions [nbeams x nbins x nens x nepochs]
    %       where nens is the number of ensembles (measurement profiles) in an
    %       epoch (observational period).
    %   option: flag to indicate whether to normalise the data prior to
    %       removing the background shear to deduct. Allowed values:
    %       option = 0, do NOT normalise
    %       option = 1, normalise [default]
    %
    % BDS, Bangor January 2016
    """
    nb, nz, nt, ne = v.shape # [nbeams x nbins x nens x nepochs].

    # deduct background shear (mean for each bin in each beam across all
    # ensembles within each epoch) to give turbulent velocity variation
    if normalize: # normalise data first if option is true
        # deduct the median value across all the bins for each beam /
        # ensemble / epoch to normalise each ensemble profile
        vp = v - np.nanmedian(v, axis=1)[:, np.newaxis, :, :] # Remove z-average.

        # now remove background shear
        vp = vp - np.nanmean(vp, axis=2)[:, :, np.newaxis, :]
    else:
        # remove backbround shear without normalising data first
        vp = v - np.nanmean(v, axis=2)[:, :, np.newaxis, :] # Remove t-average.

    return vp


def calcDLL(vp, rx=None, npts_ensavg=0, how_odd='mean_of_squared_seps'):
    """
    USAGE
    -----
    DLL, rx = calcD(vp, rx=None, npts_ensavg=0, how_odd='mean_of_squared_seps')

    The structure function DLL is DLL(x, rx).

    Input
    -----
    vp: Turbulent velocity array from a High-frequency (~1 Hz or faster) ADCP in pulse-to-pulse coherent mode with background shear removed.  Array size [nbeams x nbins x ntimestamps x nepochs]
    where nbeams is number of beams, nbins is number of
    bins per beam, ntimestamps is the number of ensembles per observational
    period (epoch) (i.e. the averging period), and nepochs is the
    number of observational periods.

    rx: [optional] Vector of separations (in bins) over which the velocity
    difference is calculated in numbers of bins.  If omitted,
    differences for all possible values is used

    npts_ensavg: [optional] Number of ensembles to average together
    after calculating the structure function from all ensembles in the given "vp".

    option: [optional] Determines which averaging approach for separations
    equivalent to an odd number of bins.  option = 'mean_of_squared_seps'
    is the mean of the squared difference [default, used if option is not specified]
    option = 'squared_mean_of_absolute_difference' is square of the mean of the
    absolute difference, and option = 'mean_of_seps_squared' is the
    square of the mean of the difference.

    OUTPUT
    ------
    DLL: Second-order structure function array with dimensions [nbeams x nbins x nepochs x nr],
    where 'nr' is the length of rx vector, i.e., the number of separation distances at which the
    structure function D is resolved.

    rx: Array of separations (in number of bins, not meters).

    ***This is a direct translation of the Matlab function 'ADCP_SF_calc_D.m' by Brian Scannell, University of Bangor, which was in turn based on original code developed by Phil Wiles, described in the following references:

    *Wiles, PJ et al (2005) A Novel Technique for Measuring Mixing Using Acoustic Doppler Current
    Profilers (ADCPs), Proceedings of the IEEE/OESEighth Working Conference on Current Measurement
    Technology, doi:10.1109/CCM.2005.1506324.

    *Wiles, PJ et al (2006), A novel technique for measuring the rate of turbulent dissipation in
    the marine environment, Geophys. Res. Lett., 33(21), 1?5, doi:10.1029/2006GL027050.

    Translated by André Palóczy and added "npts_avg option" (Jan/2019).
    apaloczy@ucsd.edu
    """
    nbeams, nbins, ntimestamps, nepochs = vp.shape
    if rx is None:
        rx = np.arange(2, nbins-1, 1)

    rx = np.int32(rx)
    if np.logical_or(rx.max()>nbins-1, rx.min()<2):
        raise ValueError('*** ERROR *** rx must be vector of integer values with maximum range 2 to nbins-2')

    nr = rx.size
    DLL = np.empty((nbeams, nbins, ntimestamps, nepochs, nr))*np.nan

    # calculate square of velocity differences.
    for ib in range(nbeams):
        for n in range(nr):
            rn = rx[n]
            if not rn%2: # This separation is an *even* number of bins.
                rnm = rn//2
                ix = np.int32(np.arange(rnm, nbins - rnm, 1))
                DLLn = (vp[ib, ix+rnm, :, :] - vp[ib, ix-rnm, :, :])**2
            else:       # This separation is an *odd* number of bins.
                rnlm, rnhm = np.floor(rn/2), np.ceil(rn/2)
                ix = np.int32(np.arange(rnhm, nbins - rnhm, 1))
                ixl = ix - int(rnhm)
                ixh = ix - int(rnlm)
                vnl = vp[ib, ixl, :, :]
                vnh = vp[ib, ixh, :, :]
                vnll = vp[ib, ixl+rn, :, :]
                vnhh = vp[ib, ixh+rn, :, :]
                if how_odd=='mean_of_squared_seps': # Mean of squared differences.
                    DLLn = 0.5*((vnll - vnl)**2 + (vnhh - vnh)**2)
                elif how_odd=='squared_mean_of_absolute_difference':
                    DLLn = (0.5*(np.abs(vnll - vnl) + np.abs(vnhh - vnh)))**2
                elif how_odd=='mean_of_seps_squared':
                    DLLn = (0.5*(vnll - vnl + vnhh - vnh))**2
                else:
                    raise NameError("Invalid treatment option '%s' for odd separations."%how_odd)

            DLL[ib, ix, :, :, n] = DLLn

    # Average the square velocities across timestamps within an ensemble average to give the
    # final structure function array.
    if npts_ensavg>0:
        DLL = blkavgens(DLL, npts_ensavg)

    return DLL, rx


def calcepsilon(DLL, rx23, CV2=2.0, method='scannell_etal2017', ret_fitcovs=False):
    """
    USAGE
    -----
    epsilon = calcepsilon(DLL, rx23, CV2=2.0, method='scannell_etal2017', ret_fitcovs=False)

    CV2: Turbulent constant in the Kolmogorov scaling.
    ret_fitcovs: Whether to return the variances of each coefficient in the polynomial fit.

    Calculate Turbulent Kinetic energy (TKE) rate of dissipation (epsilon) from the second-order
    turbulence structure function (DLL) derived from along-beam ADCP velocity measurements.

    ***This is a direct translation (with some adaptations/additions) of the Matlab function 'ADCP_SF_calc_D.m' by Brian Scannell, University of Bangor, which was in turn based on original code developed by Phil Wiles, described in the references in the original docstring below:

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %[epsilon,noise] = ADCP_SF_calc_epsilon(D,r23,CV2)
    %
    %epsilon_calc : calculates TKE dissipation (epsilon) using data from an HF
    %ADCP based on the structure function methodology.
    %Undertakes a linear regression of the structure function array, D(z,r)
    %containing the mean square turbulent velocity difference for a range of
    %separation distances, r, centred about a point, z (essentially the ADCP
    %bin).
    %Linear regression of D(z,r) against r^(2/3) undertaken for each z using
    %D(z,r) values from all available separation distances - which depends on
    %the bin.
    %
    %Input arguments:
    %   D:  The structure function array (D(z,r) of the mean of the square of
    %       turbulent velocity differences for a range of separations. The
    %       velocity backgraound shear is removed before D(z,r) is calculated.
    %       The square of the turbulent velocity difference for each separation
    %       distance is averaged across a range of observation ensembles in
    %       order to provide a statistically representative result.
    %       Array of dimensions: [nbeams x nbins x nepochs x nr] where:
    %       nbeams is the number of ADCP beams of radial velocities;
    %       nbins is the number of ADCP bins;
    %       nepochs is the number of observational periods over which the
    %       average velocity difference has been calculated; and
    %       nr is the number of separation distances over which the velocity
    %       difference has been calculated
    %   r23: Vector of separation distances to the power of (2/3) - vector
    %       length must equal nr
    %   CV2: Structure function turbulence constant
    %
    %Outputs:
    %   epsilon: Estimated TKE dissipation values. Array of dimensions:
    %       [nbeams x nbins x nepochs]
    %   noise: Estimated noise level based on extrapolated velocity difference
    %       at zero separation
    %
    %Original structure function code developed by Phil Wiles and based on
    %papers:
    %   Wiles, PJ et al (2005) A Novel Technique for Measuring Mixing Using
    %       Acoustic Doppler Current Profilers (ADCPs), Proceedings of the
    %       IEEE/OESEighth Working Conference on Current Measurement
    %       Technology, doi:10.1109/CCM.2005.1506324
    %   Wiles, PJ et al (2006), A novel technique for measuring the rate of
    %       turbulent dissipation in the marine environment, Geophys. Res.
    %       Lett., 33(21), 1?5, doi:10.1029/2006GL027050.
    """
    nbeams, nbins, nepochs, nr = DLL.shape

    # Fit line (or cubic polynomial if method=='scannell_etal2017')
    # to the measured structure function DLL(r**(2/3)).
    if method=='scannell_etal2017': # A In the cubic-polynomial case (Scannell et al., 2017),
                                    # DLL(x,r) has a quadratic dependence on 'r', since
                                    # [r**(2/3)]**3 = r**2.
        polydeg = 3 # A3*r23**(3) + A2*r23**(2) + A1*r23**(1) + A0 = DLL(r^[2/3]), or
                    # A3*r**(2) + A2*r**(4/3) + A1*r**(1) + A0 = DLL(r)
                    # A3*WAVES + ... + A1*TURBULENCE + INSTRUMENT NOISE = DLL(r).
    elif method=='standard': # A1*r23**(1) + A0 = DLL(r23)
        polydeg = 1

    # define arrays to hold values.
    coeffs = np.empty((nbeams, nbins, nepochs, polydeg + 1))*np.nan
    corrs = np.empty((nbeams, nbins, nepochs))*np.nan # Magnitudes of corr coeffs of polynomial fits.

    # Progress bar
    # h = waitbar(0,'Regression analysis')

    # regression analysis has to be done for each beam / bin / epoch separately.
    for ibm in range(nbeams):
        for ibn in range(nbins):
            for iep in range(nepochs):
                # try:
                dll = np.squeeze(DLL[ibm, ibn, iep, :])
                ix = np.isfinite(dll)
                if ix.sum()>=(polydeg+1): # Calculate lstsq fit if at least "polydeg"+1 points are available.
                    if ret_fitcovs:
                        coeffs[ibm, ibn, iep, :], polycovs = polyfit(rx23[ix], dll[ix], polydeg, full=False, cov=True)
                        corrs[ibm, ibn, iep, :] = np.diag(polycovs)
                    else:
                        coeffs[ibm, ibn, iep, :] = polyfit(rx23[ix], dll[ix], polydeg) # x, y(x), degree of polynomial.
                else:
                    coeffs[ibm, ibn, iep, :] = np.nan

    noise = coeffs[..., -1]                # A0 coeff, instrument noise.
    epsilon = (coeffs[..., -2]/CV2)**(3/2) # A1 coeff, calculate the dissipation here
                                           # (in m^2 s^-3), or (W kg^-1).
                                           # epsilon = (A1/CV2)**(3/2)

    epsilon[~np.isfinite(epsilon)] = np.nan # Gets rid of real/complex NaNs.
    epsilon[epsilon<=0] = np.nan            # Gets rid of negative dissipation estimates
                                            # (assumes to be noise).

    if ret_fitcovs:
        return epsilon, noise, corrs
    else:
        return epsilon, noise


def blkavgens(arr, npts):
    nb, nz, nt, ne, nsep = arr.shape
    NAVGS = int(np.ceil(nt/npts))
    arravg = np.empty((nb, nz, NAVGS, ne, nsep))
    fl = 0
    fr = npts
    for n in range(NAVGS):
        arravg[:, :, n, :, :] = np.nanmean(arr[:, :, fl:fr, :, :], axis=2)
        fl = fr
        fr += npts

    return arravg

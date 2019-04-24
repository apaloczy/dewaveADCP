import numpy as np
from xarray import DataArray
from scipy.optimize import curve_fit
from scipy.signal import welch
from .utils import sind, cosd
from .beam2earth import janus2xyz
from .StructureFunction import calcvp


def calc_Suiui(Bi, fs=1.0,
               window='hanning', nperseg=1024, normalize_vp=True, gaps='interpolate', **kw):
    """
    USAGE
    -----
    Su1u1, Su2u2, ..., Sunun = calc_Suiui((b1, b2, ..., bn), fs=1.0, window='hanning',
                                                             nperseg=1024,
                                                             normalize_vp=True,
                                                             gaps='interpolate', **kw)

    Returns a tuple of along-beam autospectra (Su1u1, Su2u2, ..., Sunun) from
    a tuple of along-beam velocities.
    """
    # Bi = [calcvp(bi, normalize=normalize_vp) for bi in Bi]

    # Fill NaN's with the (time) mean (minimizes side effects on the spectrum).
    # The time-mean of each bin should already be very small [O(1e-7) or smaller],
    # as it has been subtracted out by calcvp().
    Bi = list(Bi)
    if normalize_vp:
        for i in range(len(Bi)):
            Bi[i] = calcvp(Bi[i], normalize=True)

    if gaps=='mean': # Replace missing values with mean.
        for i in range(len(Bi)):
            for n in range(Bi[0].shape[0]):
                bni = Bi[i][n,:]
                fnan = np.isnan(bni)
                Bi[i][n,fnan] = np.nanmean(bni)
    elif gaps=='interpolate':
        idx = np.arange(0, Bi[0].shape[1])
        for i in range(len(Bi)):
            bi = Bi[i]
            for n in range(bi.shape[0]):
                Bi[i][n, :] = DataArray(bi[n, :], coords=dict(t=idx), dims='t').interpolate_na(dim='t', method='linear').values

    Suiui = [welch(bi, fs=fs,
                   window=window,
                   nperseg=nperseg,
                   return_onesided=True, axis=-1, **kw) for bi in Bi]

    fSuiui = Suiui[0][0] # Frequencies should be identical for all beams' autospectrum.
    Suiui = [suiui[1] for suiui in Suiui]

    return fSuiui, Suiui


def calc_Couwvw(Su1u1, Su2u2, Su3u3, Su4u4, theta):
    """
    USAGE
    -----
    Couw, Covw = calc_Couwvw(Su1u1, Su2u2, Su3u3, Su4u4, theta)

    ('Su1u1', 'Su2u2') and ('Su3u3', 'Su4u4') are two pairs of along-beam velocity autospectra,
    with the two beams in each pair being opposite from one another (e.g., beams 1-2 and 3-4 in
    TRDI's convention):

    Couw(omega) ~ Su1u1(omega) - Su2u2(omega)
    Covw(omega) ~ Su3u3(omega) - Su4u4(omega)
    """
    den = 4*sind(theta)*cosd(theta)
    Couw = (Su1u1 - Su2u2)/den
    Covw = (Su3u3 - Su4u4)/den

    return Couw, Covw


def calc_Sww(p, t, z, h, sw_disprel=False, rho=1024, fs=1.0, window='hanning', nperseg=1024, **kw):
    """
    USAGE
    -----
    f, Sww = calc_Sww(p, t, z, h, sw_disprel=True, rho=1024, fs=1.0, window='hanning',
                                  nperseg=1024, **kw)

    FIXME: Order of magnitude of the Sww returned is wrong.
    """
    omega, Spp = welch(p, fs=fs, window=window, nperseg=nperseg, return_onesided=True, **kw)
    omega = 2*np.pi*omega # Radian frequency.

    if sw_disprel:
        g = 9.81 # [m2/s2].
        k = omega/np.sqrt(g*h)
    else:
        k = ksgw(omega, h) # Get k from the linear wave dispersion relation [rad/m].

    # Convert pressure spectrum to w spectrum at each depth using linear wave theory.
    cff = (k/(rho*omega))**2
    tanhkh = np.tanh(k*(z[:,np.newaxis] + h))
    Sww = Spp*cff*tanhkh**2

    omega = omega/(2*np.pi) # Back to linear frequency.

    return omega, Sww


def calc_uwvw(Couw, Covw, k0=1e-4):
    """
    USAGE
    -----
    uw, vw = calc_uwvw(Couw, Covw, k0=1e-4)

    Integrate the uw, vw co-spectra from frequency zero up to the wave cutoff frequency
    to obtain the de-waved Reynolds stresses <uw> and <vw>.
    """
    return uw, vw


def get_kwc(cutoff_perc=0.3):
    """
    USAGE
    -----
    owc = get_kwc(cutoff_perc=0.3)

    Calculate the wave cut-off wavenumber, defined
    as the frequency where the Sww autospectrum
    reaches 'cutoff_perc' percent of the along-beam
    velocity autospectrum, Suiui.
    """
    raise NotImplementedError


def Somega2Sk(omega, b1, b2, b3, b4, theta):
    """
    USAGE
    -----
    k = Somega2Sk(omega, b1, b2, b3, b4, theta)

    Converts a frequency spectrum 'Sxx(omega)' to
    a wavenumber spectrum 'Sxx(k)' by assuming Taylor's
    or "Frozen Flow" Hypothesis.
    """
    # Use the burst-mean velocity profile.
    b1 = np.nanmean(b1, axis=1)[:, np.newaxis]
    b2 = np.nanmean(b2, axis=1)[:, np.newaxis]
    b3 = np.nanmean(b3, axis=1)[:, np.newaxis]
    b4 = np.nanmean(b4, axis=1)[:, np.newaxis]

    Vx, Vy, _ = janus2xyz(b1, b2, b3, b4, theta)
    U = np.sqrt(Vx**2 + Vy**2)
    k = omega[np.newaxis,:]/U # omega = k*|U|.

    return k


def kamel72spec(k):
    """Free parameters: c1, c2 = (uwstar, k0)."""
    def fkamel72spec(k, c1, c2):
        f73 = 7/3
        C = f73*np.sin(np.pi/f73)/np.pi

        return c1*C*(1/c2)/(1 + (k/c2)**f73)

    return fkamel72spec


def Couwvw_fit(k, Suiuik, func=kamel72spec, c1c2guess=(5e-4, 2)):
    """..."""
    c1c2, _ = curve_fit(func, k, Suiuik, p0=c1c2guess, maxfev=10000)
    ustar, k0 = c1c2
    uwvw_model = kamel72spec(k, ustar, k0)

    return uwvw_model, ustar, k0


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
    nz, nt = v.shape # [nbins x nens].

    # deduct background shear (mean for each bin in each beam across all
    # ensembles within each epoch) to give turbulent velocity variation
    if normalize: # normalise data first if option is true
        # deduct the median value across all the bins for each beam /
        # ensemble / epoch to normalise each ensemble profile
        vp = v - np.nanmedian(v, axis=0) # Remove z-average.

        # now remove background shear
        vp = vp - np.nanmean(vp, axis=1)[..., np.newaxis]
    else: # remove backbround shear without normalising data first
        vp = v - np.nanmean(v[np.newaxis, ...], axis=1) # Remove t-average.

    return vp


def ksgw(omega, h):
	"""
	The code below was translated to Python by André Palóczy
	from the original Matlab code "ib09_get_wavenumber.m"
	by Falk Feddersen.

	# % Falk Feddersen (c) 2001
	# %
	# % function that takes the radian wave frequency and
	# % a vector of depths and returns the wavenumber at that
	# % depth by solving the dispersion relationship
	# %
	# % function k = get_wavenum(omega,h)

	# function k = ib09_get_wavenumber(omega,h)

	# % returns the wavenumber of the gravity wave
	# % dispersion relation, by using newtons method

	# % the initial guess will be the shallow water wavenumber
	"""
	g = 9.81 # [m/s2].

	k = omega/np.sqrt(g*h) # Shallow water wavenumber.
	f = g*k*np.tanh(k*h) - omega**2

	while np.max(np.abs(f))>1e-10:
		dfdk = g*k*h*(1./np.cosh(k*h))**2 + g*np.tanh(k*h)
		k = k - f/dfdk
		f = g*k*np.tanh(k*h) - omega**2

	return k

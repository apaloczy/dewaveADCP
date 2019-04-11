import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch
from .utils import sind, cosd
from .beam2earth import janus2xyz


def calc_Suiui(Bi, fs=1.0, window='hanning', nperseg=1024, normalize_vp=True, **kw):
    """
    USAGE
    -----
    Su1u1, Su2u2, ..., Sunun = calc_Suiui((b1, b2, ..., bn), fs=1.0, window='hanning',
                                                             nperseg=1024,
                                                             normalize_vp=True, **kw)

    Returns a tuple of along-beam autospectra (Su1u1, Su2u2, ..., Sunun) from
    a tuple of along-beam velocities.
    """
    # Bi = [calcvp(bi, normalize=normalize_vp) for bi in Bi]

    # Fill NaN's with the (time) mean (minimizes side effects on the spectrum).
    # The time-mean of each bin should already be very small [O(1e-7) or smaller],
    # as it has been subtracted out by calcvp().
    for i in range(len(Bi)):
        for n in range(Bi[0].shape[0]):
            bni = Bi[i][n,:]
            fnan = np.isnan(bni)
            Bi[i][n,fnan] = np.nanmean(bni)

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


def calc_Sww(z, h, Spp):
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
    Sww = Spp*cff*np.tanh(k*(z + h))**2

    return Sww

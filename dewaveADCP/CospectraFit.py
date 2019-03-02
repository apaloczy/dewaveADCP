import numpy as np
from scipy.optimize import least_squares, curve_fit
# from .utils import sind, cosd
from utils import sind, cosd


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
    cff = (k/(rho*omega))**2
    Sww = Spp*cff*np.tanh(k*(z + h))**2

    return Sww

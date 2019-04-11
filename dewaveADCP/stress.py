# Functions for calculating velocity covariances (Reynolds stresses) from a moored ADCP.
import numpy as np
from VarianceFit import varfitw, sgwvar_func, calc_beta
from VerticalDetrend import dewave_verticaldetrend
from utils import sind, cosd
from ap_tools.fit import fourfilt

d2r = np.pi/180

######################
#### 5-beam Janus ####
######################

def rstress5(b1, b2, b3, b4, b5, theta, phi1, phi2, phi3):
    """
    Calculate components of the Reynolds stress tensor
    from along-beam velocities measured with a 4-beam Janus ADCP.
    """
    raise NotImplementedError


def uwrs5(b1, b2, b5, uv, theta, phi2, phi3):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4, b5.

    The formula for the small-angle
    approximation is used (D&S equation 132).
    """
    b1var, b2var, b5var = map(bvar, (b1, b2, b5))
    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S4C2 = (Sth**4)*(Cth**2)

    phi2, phi3 = phi2*d2r, phi3*d2r
    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(4*S6C2)

    # Dewey & Stringer (2007)'s equation (133).
    uw = coeff*(S5C1*b2mb1 + 2*S4C2*phi2*b2pb1 - 4*S4C2*phi3*b5var - 4*S4C2*phi3*uv)

    return uw


def vwrs5(b3, b4, b5, uv, theta, phi2, phi3):
    """
    Calculates the <v'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4, b5.

    The formula for the small-angle
    approximation is used (D&S equation 133).
    """
    b3var, b4var, b5var = map(bvar, (b3, b4, b5))
    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S4C2 = (Sth**4)*(Cth**2)

    phi2, phi3 = phi2*d2r, phi3*d2r
    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(4*S6C2)

    # Dewey & Stringer (2007)'s equation (133).
    vw = coeff*(S5C1*b4mb3 - 2*S4C2*phi2*b4pb3 + 4*S4C2*phi3*b5var + 4*S4C2*phi3*uv)

    return vw


def uurs5(b1, b2, b5, theta, phi3):
    raise NotImplementedError


def vvrs5(b1, b2, b3, b4, b5, theta, phi2, phi3):
    raise NotImplementedError


def wwrs5(b1, b2, b3, b4, b5, theta, phi2, phi3):
    raise NotImplementedError

######################
#### 4-beam Janus ####
######################


def rstress4(b1, b2, b3, b4, b5, theta, phi1, phi2, phi3):
    """
    Calculate components of the Reynolds stress tensor
    from along-beam velocities measured with a 4-beam Janus ADCP.
    """
    raise NotImplementedError


def uwrs4(b1, b2, theta, phi2, phi3, enu=None, averaged=True):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <u'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 32).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r
    b1var, b2var = map(bvar, (b1, b2))

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        u, v, w = enu
        uv = u*v
        ww = w*w
    else:
        ww = uv = b1var*0

    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32).
    uw = -(coeff*b2mb1 + (b2pb1/2 - ww)*phi3/S2 - phi2*uv)

    if averaged:
        uw = np.nanmean(uw, axis=1)

    return uw


def vwrs4(b3, b4, theta, phi2, phi3, enu=None, averaged=True):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <v'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 33).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r
    b3var, b4var = map(bvar, (b3, b4))

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        raise NotImplementedError
    else:
        ww = uv = b3var*0

    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32).
    vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)

    if averaged:
        vw = np.nanmean(vw, axis=1)

    return vw


def uwrs4_detrend(b1, b2, r, theta, phi2, phi3, enu=None, averaged=True, dpoly=1, detrend_time=False, lowhi_Tcutoff=None, dts=None, **kw):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <u'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 32).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2
    phi2, phi3 = phi2*d2r, phi3*d2r
    nz, nt = b1.shape

    # remove wave signal by removing a polynomial fit in the vertical.
    if lowhi_Tcutoff is None:
        b1 = dewave_verticaldetrend(b1, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
        b2 = dewave_verticaldetrend(b2, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
    else:
        b1hi = b1*np.nan
        b2hi = b1hi.copy()
        b1lo = b1hi.copy()
        b2lo = b1hi.copy()
        for k in range(nz):
            b1k, b2k = b1[k,:], b2[k,:]
            b1k[np.isnan(b1k)] = np.nanmean(b1k)
            b2k[np.isnan(b2k)] = np.nanmean(b2k)
            b1hi[k,:] = fourfilt(b1k, dts, lowhi_Tcutoff, dts/2)
            b2hi[k,:] = fourfilt(b2k, dts, lowhi_Tcutoff, dts/2)
            b1lo[k,:] = fourfilt(b1k, dts, nt*dts, lowhi_Tcutoff)
            b2lo[k,:] = fourfilt(b2k, dts, nt*dts, lowhi_Tcutoff)

        # Detrend only high-passed series to remove waves.
        b1hi = dewave_verticaldetrend(b1hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
        b2hi = dewave_verticaldetrend(b2hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)

        # Add back to low-passed part to reconstruct the full time series.
        b1 = b1lo + b1hi
        b2 = b2lo + b2hi

    b1var, b2var = map(bvar, (b1, b2))

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        raise NotImplementedError
    else:
        ww = uv = b1var*0

    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32).
    uw = -(coeff*b2mb1 + (b2pb1/2 - ww)*phi3/S2 - phi2*uv)

    if averaged:
        uw = np.nanmean(uw, axis=1)

    return uw


def vwrs4_detrend(b3, b4, r, theta, phi2, phi3, enu=None, averaged=True, dpoly=1, detrend_time=False, lowhi_Tcutoff=None, dts=None, **kw):
    """
    Calculates the <v'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <v'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 33).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2
    phi2, phi3 = phi2*d2r, phi3*d2r
    nz, nt = b3.shape

    # remove wave signal by removing a polynomial fit in the vertical.
    if lowhi_Tcutoff is None: # Vertically detrend total signal.
        b3 = dewave_verticaldetrend(b3, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
        b4 = dewave_verticaldetrend(b4, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
    else: # Vertically detrend only high-passed signal.
        b3hi = b3*np.nan
        b4hi = b3hi.copy()
        b3lo = b3hi.copy()
        b4lo = b3hi.copy()
        for k in range(nz):
            b3k, b4k = b3[k,:], b4[k,:]
            b3k[np.isnan(b3k)] = np.nanmean(b3k)
            b4k[np.isnan(b4k)] = np.nanmean(b4k)
            b3hi[k,:] = fourfilt(b3k, dts, lowhi_Tcutoff, dts/2)
            b4hi[k,:] = fourfilt(b4k, dts, lowhi_Tcutoff, dts/2)
            b3lo[k,:] = fourfilt(b3k, dts, nt*dts, lowhi_Tcutoff)
            b4lo[k,:] = fourfilt(b4k, dts, nt*dts, lowhi_Tcutoff)

        # Detrend only high-passed series to remove waves.
        b3hi = dewave_verticaldetrend(b3hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
        b4hi = dewave_verticaldetrend(b4hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)

        # Add back to low-passed part to reconstruct the full time series.
        b3 = b3lo + b3hi
        b4 = b4lo + b4hi

    b3var, b4var = map(bvar, (b3, b4))

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        raise NotImplementedError
    else:
        ww = uv = b3var*0

    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (33).
    vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)

    if averaged:
        vw = np.nanmean(vw, axis=1)

    return vw


def uwrs4_varfit(b1, b2, r, theta, phi2, phi3, h, alpha, sep, enu=None, averaged=True, c1c2guess=(0.2, 0.4)):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <u'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 32).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r
    b1var, b2var = map(bvar, (b1, b2))
    b1var = np.nanmean(b1var, axis=1)
    b2var = np.nanmean(b2var, axis=1)

    # remove wave signal using the variance fit method
    # (Whipple et al., 2006; Rosman et al., 2008)
    fsgw = sgwvar_func(h, theta, alpha)
    b1varw = varfitw(r, b1var, fsgw, c1c2guess=c1c2guess)
    b2varw = varfitw(r, b2var, fsgw, c1c2guess=c1c2guess)

    # Calculate 'beta' from the fitted along-beam wave variance
    # profiles and a specified separation.
    beta1, _ = calc_beta(b1varw, sep)
    beta2, zidx = calc_beta(b2varw, sep)
    beta12 = 0.5*(beta1 + beta2)
    b2var = b2var[zidx] # Cap variance profiles to depths where RS is available.
    b1var = b1var[zidx]

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        raise NotImplementedError
    else:
        ww = uv = b1var*0

    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32), divided by (1 + beta**2).
    uw = -(coeff*b2mb1 + (b2pb1/2 - ww)*phi3/S2 - phi2*uv)/(1 + beta12**2)

    if averaged:
        uw = np.nanmean(uw, axis=1)

    return uw


def vwrs4_varfit(b3, b4, r, theta, phi2, phi3, h, alpha, sep, enu=None, averaged=True, c1c2guess=(0.2, 0.4)):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <v'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 33).
    """
    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r
    b3var, b4var = map(bvar, (b3, b4))
    b3var = np.nanmean(b3var, axis=1)
    b4var = np.nanmean(b4var, axis=1)

    # remove wave signal using the variance fit method
    # (Whipple et al., 2006; Rosman et al., 2008)
    fsgw = sgwvar_func(h, theta, alpha)
    b3varw = varfitw(r, b3var, fsgw, c1c2guess=c1c2guess)
    b4varw = varfitw(r, b4var, fsgw, c1c2guess=c1c2guess)

    # Calculate 'beta' from the fitted along-beam wave variance
    # profiles and a specified separation.
    beta3, _ = calc_beta(b3varw, sep)
    beta4, zidx = calc_beta(b4varw, sep)
    beta34 = 0.5*(beta3 + beta4)
    b4var = b4var[zidx] # Cap variance profiles to depths where RS is available.
    b3var = b3var[zidx]

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        u, v, w = enu
        uv = u*v
        ww = w*w
    else:
        ww = uv = b3var*0

    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (33), divided by (1 + beta**2).
    vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)/(1 + beta34**2)

    if averaged:
        vw = np.nanmean(vw, axis=1)

    return vw


#############################
#### ancillary functions ####
#############################

def bvar(b):
    """
    Calculates the time series of profiles of variance of along-beam velocity
    from the along-beam velocities 'b'.
    """
    return (b - np.nanmean(b, axis=1)[:, np.newaxis])**2


def rot_uwvw(uw, vw, phi1):
    """
    Rotates the vector of vertical transport of horizontal momentum (uw + i*vw)
    to Earth coordinates (vertical transport of *eastward* and *meridional* momentum).
    """
    Sphi1, Cphi1 = sind(phi), cosd(phi1)
    R11, R12 = +u*Cphi1, +v*Sphi1
    R21, R22 = -u*Sphi1, +v*Cphi1

    R11, R12 = R11[np.newaxis, np.newaxis, :], R12[np.newaxis, np.newaxis, :]
    R21, R22 = R21[np.newaxis, np.newaxis, :], R22[np.newaxis, np.newaxis, :]

    # Time-dependent rotation matrix.
    R = np.matrix([[R11, R12],
                   [R21, R22]])

    nz, nt = uw.shape
    uwrot = uw*np.nan
    vwrot = vw*np.nan
    for n in range(nt):
        Rn = R[..., n]
        for k in range(nz):
            uwvw = np.matmul(Rn, np.matrix([uw[:, n], vw[:, n]]).T) # 2x1 vector on LHS.
            uwrot[:, n] = uwvwr[0]
            vwrot[:, n] = uwvwr[1]

    return uwrot, vwrot
# Functions for calculating velocity covariances (Reynolds stresses) from a moored ADCP.
import numpy as np
from scipy.optimize import curve_fit
from xarray import DataArray
from .VarianceFit import varfitw, sgwvar_func, calc_beta
from .VerticalDetrend import dewave_verticaldetrend, fexp
from .AdaptiveFiltering import bvarAF
from .utils import sind, cosd, fourfilt

d2r = np.pi/180

######################
#### 5-beam Janus ####
######################

def rstress5(b1, b2, b3, b4, b5, theta, phi2, phi3, dewave=True,
             uv=None, averaged=True, enslen=None, z=None, t=None, **kw):
    """
    USAGE
    -----
    uw, vw, uu, vv, ww, tke, aniso = rstress5(b1, b2, b3, b4, b5, theta, phi2, phi3, dewave=True,
                                              uv=None, averaged=True, enslen=None, z=None, t=None,
                                              sep=6, Lw=128, max_badfrac=0.3, verbose=False)

    Calculate components of the Reynolds stress tensor, the turbulent kinetic energy and the anisotropy ratio
    from along-beam velocities measured with a 5-beam Janus ADCP.

    If 'dewave' is set to True (default), uses the Adaptive Filtering Method to filter out the surface
    gravity wave signal from beam velocities prior to calculating the stresses.
    """
    return NotImplementedError


def uwrs5(b1, b2, b5, theta, phi2, phi3, variances=True, uv=None, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4, b5.

    The formula for the small-angle
    approximation is used (D&S equation 132).
    """
    if variances:
        b1var, b2var, b5var = b1, b2, b5
    else:
        b1var, b2var, b5var = map(bvar, (b1, b2, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S4C2 = (Sth**4)*(Cth**2)

    phi2, phi3 = phi2*d2r, phi3*d2r
    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(4*S6C2)

    if uv is None:
        uv = b1var*0

    # Dewey & Stringer (2007)'s equation (132).
    if tilt_corr:
        uw = coeff*(S5C1*b2mb1 + 2*S4C2*phi3*b2pb1 - 4*S4C2*phi3*b5var - 4*S6C2*phi2*uv)
    else:
        uw = coeff*S5C1*b2mb1

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            uw = DataArray(uw, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            uw = np.nanmean(uw, axis=1)

    return uw


def vwrs5(b3, b4, b5, theta, phi2, phi3, variances=True, uv=None, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the <v'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4, b5.

    The formula for the small-angle
    approximation is used (D&S equation 133).
    """
    if variances:
        b3var, b4var, b5var = b3, b4, b5
    else:
        b3var, b4var, b5var = map(bvar, (b3, b4, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S4C2 = (Sth**4)*(Cth**2)
    S4C4 = (Sth*Cth)**4

    phi2, phi3 = phi2*d2r, phi3*d2r
    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(4*S6C2)

    if uv is None:
        uv = b3var*0

    # Dewey & Stringer (2007)'s equation (133).
    if tilt_corr:
        vw = coeff*(S5C1*b4mb3 - 2*S4C2*phi3*b4pb3 + 4*S4C4*phi3*b5var + 4*S6C2*phi2*b5var + 4*S6C2*phi3*uv)
    else:
        vw = coeff*S5C1*b4mb3

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            vw = DataArray(vw, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            vw = np.nanmean(vw, axis=1)

    return vw


def uurs5(b1, b2, b5, theta, phi3, variances=True, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the <u'u'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2, b5.

    The formula for the small-angle
    approximation is used (D&S equation 129).
    """
    if variances:
        b1var, b2var, b5var = b1, b2, b5
    else:
        b1var, b2var, b5var = map(bvar, (b1, b2, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S4C2 = (Sth**4)*(Cth**2)
    C2 = Cth**2

    phi3 = phi3*d2r
    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = -1/(4*S6C2)

    # D&S Equation 129.
    if tilt_corr:
        uu = coeff*(-2*S4C2*(b2pb1 - 2*C2*b5var) + 2*S5C1*phi3*b2mb1)
    else:
        uu = coeff*(-2*S4C2*(b2pb1 - 2*C2*b5var))

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            uu = DataArray(uu, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            uu = np.nanmean(uu, axis=1)

    return uu


def vvrs5(b1, b2, b3, b4, b5, theta, phi2, phi3, variances=True, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the <v'v'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2, b3, b4 and b5.

    The formula for the small-angle
    approximation is used (D&S equation 130).
    """
    if variances:
        b1var, b2var, b3var, b4var, b5var = b1, b2, b3, b4, b5
    else:
        b1var, b2var, b3var, b4var, b5var = map(bvar, (b1, b2, b3, b4, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)
    S3C3 = (Sth*Cth)**3
    S4C2 = (Sth**4)*(Cth**2)
    C2 = Cth**2

    phi2, phi3 = phi2*d2r, phi3*d2r
    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = -1/(4*S6C2)

    # D&S Equation 130.
    if tilt_corr:
        vv = coeff*(-2*S4C2*(b4pb3 - 2*C2*b5var) - 2*S4C2*phi3*b2pb1 + 4*S3C3*phi3*b2mb1 - 2*S5C1*phi2*b4mb3)
    else:
        vv = coeff*(-2*S4C2*(b4pb3 - 2*C2*b5var))

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            vv = DataArray(vv, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            vv = np.nanmean(vv, axis=1)

    return vv


def wwrs5(b1, b2, b3, b4, b5, theta, phi2, phi3, variances=True, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the <w'w'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2, b3, b4 and b5.

    The formula for the small-angle
    approximation is used (D&S equation 131).
    """
    if variances:
        b1var, b2var, b3var, b4var, b5var = b1, b2, b3, b4, b5
    else:
        b1var, b2var, b3var, b4var, b5var = map(bvar, (b1, b2, b3, b4, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S6C2 = (Sth**6)*(Cth**2)
    S5C1 = (Sth**5)*(Cth)

    phi2, phi3 = phi2*d2r, phi3*d2r
    b2mb1 = b2var - b1var
    b4mb3 = b4var - b3var
    coeff = -1/(4*S6C2)

    # D&S Equation 131.
    if tilt_corr:
        ww = coeff*(-2*S5C1*phi3*b2mb1 + 2*S5C1*phi2*b4mb3 - 4*S6C2*b5var)
    else:
        ww = coeff*(-4*S6C2*b5var)

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            ww = DataArray(ww, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            ww = np.nanmean(ww, axis=1)

    return ww


def tke5(b1, b2, b3, b4, b5, theta, phi3, variances=True, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the turbulent kinetic energy q^2/2
    from the along-beam velocities b1, b2, b3, b4 and b5.

    The formula for the small-angle
    approximation is used (D&S equation 134).
    """
    if variances:
        b1var, b2var, b3var, b4var, b5var = b1, b2, b3, b4, b5
    else:
        b1var, b2var, b3var, b4var, b5var = map(bvar, (b1, b2, b3, b4, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2
    C2 = Cth**2
    cotth = Cth/Sth

    phi3 = phi3*d2r
    b2mb1 = b2var - b1var
    b1234 = b1var + b2var + b3var + b4var
    coeff = 1/(4*S2)

    # D&S Equation 134.
    if tilt_corr:
        q2 = coeff*(b1234 - 2*(2*C2 - S2)*b5var - (cotth - 1)*phi3*b2mb1) # q^2/2, not q^2.
    else:
        q2 = coeff*(b1234 - 2*(2*C2 - S2)*b5var)

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            q2 = DataArray(q2, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            q2 = np.nanmean(q2, axis=1)

    return q2


def aniso_ratio(b1, b2, b3, b4, b5, theta, phi2, phi3, variances=True, tilt_corr=True, averaged=True, enslen=None, z=None, t=None):
    """
    Calculates the anisotropy ratio alpha
    from the along-beam velocities b1, b2, b3, b4 and b5.

    The formula for the small-angle
    approximation is used (D&S equation 135).
    """
    if variances:
        b1var, b2var, b3var, b4var, b5var = b1, b2, b3, b4, b5
    else:
        b1var, b2var, b3var, b4var, b5var = map(bvar, (b1, b2, b3, b4, b5))

    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2
    C2 = Cth**2
    csc2th = 1/sind(2*theta)
    Tth = Sth/Cth

    phi2, phi3 = phi2*d2r, phi3*d2r
    b2mb1 = b2var - b1var
    b4mb3 = b4var - b3var
    b1234 = b1var + b2var + b3var + b4var
    Fth = Tth*phi2*b4mb3 + (1 - 2*csc2th)*phi3*b2mb1

    # D&S Equation 135.
    if tilt_corr:
        num = 2*S2*b5var + Tth*phi3*b2mb1 - Tth*phi2*b4mb3
        den = b1234 - 4*C2*b5var + Fth
    else:
        num = 2*S2*b5var
        den = b1234 - 4*C2*b5var

    alpha = num/den

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            assert t is not None, "Need t for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            alpha = DataArray(alpha, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            alpha = np.nanmean(alpha, axis=1)

    return alpha


######################
#### 4-beam Janus ####
######################


def rstress4(b1, b2, b3, b4, b5, theta, phi1, phi2, phi3):
    """
    Calculate components of the Reynolds stress tensor
    from along-beam velocities measured with a 4-beam Janus ADCP.
    """
    raise NotImplementedError


def uwrs4(b1, b2, theta, phi2, phi3, variances=True, enu=None, tilt_corr=True, averaged=True, enslen=None, z=None):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b1, b2. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <u'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 32).
    """
    if variances:
        b1var, b2var = b1, b2
    else:
        b1var, b2var = map(bvar, (b1, b2))

    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        u, v, w = enu
        uv = (u - np.nanmean(u, axis=1)[:, np.newaxis])*(v - np.nanmean(v, axis=1)[:, np.newaxis])
        ww = (w - np.nanmean(w, axis=1)[:, np.newaxis])**2
    else:
        ww = uv = b1var*0

    b2mb1 = b2var - b1var
    b2pb1 = b2var + b1var
    coeff = 1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32).
    if tilt_corr:
        uw = -(coeff*b2mb1 + (b2pb1/2 - ww)*phi3/S2 - phi2*uv)
    else:
        uw = -coeff*b2mb1

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            uw = DataArray(uw, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            uw = np.nanmean(uw, axis=1)

    return uw


def vwrs4(b3, b4, theta, phi2, phi3, variances=True, enu=None, tilt_corr=True, averaged=True, enslen=None, z=None):
    """
    Calculates the <u'w'> component of the Reynolds stress tensor
    from the along-beam velocities b3, b4. If 'enu' is provided as a tuple
    of (u, v, w) velocities in Earth coordinates, the correction terms <w'w'> and
    <u'v'> are calculated and included in the calculation of <v'w'>. If 'enu' is
    not provided, the correction terms are neglected (default).

    The formula for the small-angle
    approximation is used (D&S equation 33).
    """
    if variances:
        b3var, b4var = b3, b4
    else:
        b3var, b4var = map(bvar, (b3, b4))

    Sth, Cth = sind(theta), cosd(theta)
    S2 = Sth**2

    phi2, phi3 = phi2*d2r, phi3*d2r

    # Calculate correction terms uv and ww from Earth velocities, if available.
    if enu is not None:
        u, v, w = enu
        uv = (u - np.nanmean(u, axis=1)[:, np.newaxis])*(v - np.nanmean(v, axis=1)[:, np.newaxis])
        ww = (w - np.nanmean(w, axis=1)[:, np.newaxis])**2
    else:
        ww = uv = b3var*0

    b4mb3 = b4var - b3var
    b4pb3 = b4var + b3var
    coeff = 1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32).
    if tilt_corr:
        vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)
    else:
        vw = -coeff*b4mb3

    if averaged:
        if enslen is not None:
            assert z is not None, "Need z for ensemble averaging."
            dims = ('z', 't')
            coords = dict(z=z, t=t)
            vw = DataArray(vw, coords=coords, dims=dims).resample(dict(t=enslen)).reduce(np.nanmean, dim='t')
        else:
            vw = np.nanmean(vw, axis=1)

    return vw


def uwrs4_detrend(b1, b2, r, theta, phi2, phi3, enu=None, averaged=True, dpoly=1, detrend_time=False, lowhi_Tcutoff=None, dts=None, cap_band=False, **kw):
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
    else:  # Vertically detrend only high-passed or band-passed signal
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

        if cap_band:
            b1 = b1lo
            b2 = b2lo
        else:
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
    coeff = 1/(2*sind(2*theta))

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
    else: # Vertically detrend only high-passed or band-passed signal.
        b3hi = b3*np.nan
        b4hi = b3hi.copy()
        b3lo = b3hi.copy()
        b4lo = b3hi.copy()
        if isinstance(lowhi_Tcutoff, tuple):
            bandpass = True
            b3bd = b3hi.copy()
            b4bd = b3hi.copy()
            Tmin, Tmax = lowhi_Tcutoff
            lowhi_Tcutoff = Tmin
        else:
            bandpass = False

        for k in range(nz):
            b3k, b4k = b3[k,:], b4[k,:]
            b3k[np.isnan(b3k)] = np.nanmean(b3k)
            b4k[np.isnan(b4k)] = np.nanmean(b4k)

            if bandpass: # Detrend a band-passed series to remove waves.
                b3bd[k,:] = fourfilt(b3k, dts, Tmax, Tmin)
                b4bd[k,:] = fourfilt(b4k, dts, Tmax, Tmin)
            else: # Detrend only high-passed series to remove waves.
                b3hi[k,:] = fourfilt(b3k, dts, lowhi_Tcutoff, dts/2)
                b4hi[k,:] = fourfilt(b4k, dts, lowhi_Tcutoff, dts/2)
                b3lo[k,:] = fourfilt(b3k, dts, nt*dts*2, lowhi_Tcutoff)
                b4lo[k,:] = fourfilt(b4k, dts, nt*dts*2, lowhi_Tcutoff)
                b3hi = dewave_verticaldetrend(b3hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)
                b4hi = dewave_verticaldetrend(b4hi, r, theta, phi2, phi3, dpoly=dpoly, detrend_time=detrend_time, **kw)

        if bandpass: # Subtract band-passed signal from total.
            b3 = b3 - b3bd
            b4 = b4 - b4bd
        else: # Add back detrended high-passed signal.
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
    coeff = 1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (33).
    vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)

    if averaged:
        vw = np.nanmean(vw, axis=1)

    return vw


def uwrs4_varfit(b1, b2, r, theta, phi2, phi3, h, alpha, sep, enu=None, c1c2guess=(0.2, 0.4)):
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

    # remove wave signal using the variance fit method
    # (Whipple et al., 2006; Rosman et al., 2008)
    print("Using guess c1, c2 = %f, %f for exp fit."%c1c2guess)
    fsgw = sgwvar_func(h, theta, alpha)
    b1varw = b1var*np.nan
    b2varw = b1varw.copy()
    for i in range(phi2.size):
        b1varw[:,i] = varfitw(r, b1var[:,i], fsgw, c1c2guess=c1c2guess)
        b2varw[:,i] = varfitw(r, b2var[:,i], fsgw, c1c2guess=c1c2guess)

    # Calculate 'beta' from the fitted along-beam wave variance
    # profiles and a specified separation.
    b1varw = np.nanmean(b1varw, axis=1)
    b2varw = np.nanmean(b2varw, axis=1)
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

    b1var = np.nanmean(b1var, axis=1)
    b2var = np.nanmean(b2var, axis=1)

    b2mb1 = b2var - b1var
    # b2pb1 = b2var + b1var
    coeff = 1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (32), divided by (1 + beta**2).
    # uw = -(coeff*b2mb1 + (b2pb1/2 - ww)*phi3/S2 - phi2*uv)/(1 + beta12**2)
    uw = -(coeff*b2mb1/(1 + beta12**2))

    return uw


def vwrs4_varfit(b3, b4, r, theta, phi2, phi3, h, alpha, sep, enu=None, c1c2guess=(0.2, 0.4)):
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

    # remove wave signal using the variance fit method
    # (Whipple et al., 2006; Rosman et al., 2008)
    print("Using guess c1, c2 = %.5f, %.5f for exp fit."%(c1c2guess[0], c1c2guess[1]))
    # print(c1c2guess[0], c1c2guess[1])
    fsgw = sgwvar_func(h, theta, alpha)
    b3varw = b3var*np.nan
    b4varw = b3varw.copy()
    for i in range(phi2.size):
        b3varw[:,i] = varfitw(r, b3var[:,i], fsgw, c1c2guess=c1c2guess)
        b4varw[:,i] = varfitw(r, b4var[:,i], fsgw, c1c2guess=c1c2guess)

    # Calculate 'beta' from the fitted along-beam wave variance
    # profiles and a specified separation.
    b3varw = np.nanmean(b3varw, axis=1)
    b4varw = np.nanmean(b4varw, axis=1)
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

    b3var = np.nanmean(b3var, axis=1)
    b4var = np.nanmean(b4var, axis=1)

    b4mb3 = b4var - b3var
    # b4pb3 = b4var + b3var
    coeff = 1/(2*sind(2*theta))

    # Dewey & Stringer (2007)'s equation (33), divided by (1 + beta**2).
    # vw = -(coeff*b4mb3 - (b4pb3/2 - ww)*phi2/S2 + phi3*uv)/(1 + beta34**2)
    vw = -(coeff*b4mb3/(1 + beta34**2))

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


def avgensuwvw(uw, vw, t, enslen, verbose=True):
    """
    Averages arrays 'uw' and 'vw' of shape (z,t) in
    ensembles of 'enslen' pings each.
    """
    nz, nt = uw.shape
    nens = int(nt//enslen)

    uwens = np.empty((nz, nens))*np.nan
    vwens = uwens.copy()
    tens = np.array([])

    il = 0
    ir = enslen
    for n in range(nens):
        if verbose:
            print("Ensemble ",n+1," / ",nens)

        uwens[:, n] = np.nanmean(uw[:,il:ir], axis=1)
        vwens[:, n] = np.nanmean(vw[:,il:ir], axis=1)
        tens = np.append(tens, t[(il+ir)//2])

        il = ir
        ir += enslen
        if ir>nt: # Last ensemble might have less pings than the others.
            ir = nt

    return uwens, vwens, tens


def rot_uwvw(uw, vw, phi1):
    """
    Rotates the vector of vertical transport of horizontal momentum (uw + i*vw)
    to Earth coordinates (vertical transport of *eastward* and *meridional* momentum).
    """
    Sphi1, Cphi1 = sind(phi1)[np.newaxis,...], cosd(phi1)[np.newaxis,...]
    uwr = uw*np.nan
    vwr = uwr.copy()

    uwr = +uw*Cphi1 + vw*Sphi1
    vwr = -uw*Sphi1 + vw*Cphi1

    return uwr, vwr

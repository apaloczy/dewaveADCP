import numpy as np
from numpy.linalg import lstsq, solve


def lstsqfit(d, r, n=1):
    """
    USAGE
    -----
    Gm = lstsq(d, r, n=1)

    Fit an 'n'-th order polynomial to the input data vector by solving an overdetermined least-squares problem of degree 'n' (default 1).
    """
    d = np.matrix(np.array(d)).T
    nd = d.size
    if n>nd:
        Error("Polynomial degree ('%d') is larger than number of points ('%d')"%(n,nd))

    # Set up data kernel matrix G and solve.
    G = np.matrix(np.array([], ndmin=2))

    for pw in range(n, 0, -1):
        mm = np.matrix(r**pw).T
        if pw==n:
            G = mm
        else:
            G = np.hstack((G, mm))

    col1s = np.matrix(np.ones((nd, 1)))
    G = np.hstack((G, col1s))

    GT = G.T
    m = solve(GT*G, GT*d) # m = (GT*G).I*GT*d.T

    return G*m


def fourfilt(x, dts, Tmax, Tmin):
    # See the docstring of the original MATLAB function.
    # The usage is the same, see the docstring for
    # low-pass, high-pass and band-pass filter examples.
    #
    # In the Python version, the variable names
    # have been changed as follows:
    #
    # MATLAB      Python
    #  delt   ->   dts
    #  tmax   ->   Tmax
    #  tmin   ->   Tmin
    # filtdat ->   xfilt
    #
    # Translated from the MATLAB function fourfilt.m
    # by André Palóczy (github.com/apaloczy).
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # function [filtdat]=fourfilt(x,delt,tmax,tmin)
    #  FOURFILT  Fourier low, high, or bandpass filter.
    #
    #     [filtdat]=fourfilt(x,delt,tmax,tmin)
    #
    #     where:   x:      data series to be filtered
    #              delt:   sampling interval
    #              tmax:   maximum period filter cutoff
    #              tmin:   minimum period filter cutoff
    #
    #     usage:  lowpassdata=fourfilt(data,0.5,2000,20)
    #
    #               gives lowpass filter with cutoff at 20.0 sec
    #               tmax set > (length(x)*delt) for no cutoff at low freq end
    #
    #     usage:  highpassdata=fourfilt(x,0.5,20,0.9)
    #
    #               gives highpass filter with cutoff at 20.0 sec
    #               tmin set < (2*delt) for no cutoff at high freq end
    #
    #     usage:  bandpassdata=fourfilt(x,0.5,20,10)
    #
    #               gives bandpass filter passing 10-20 sec. band
    #
    # Reference:
    # Walters, R. A.  and Heston, C., 1982. Removing the tidal-period variations from time-series
    # data using low-pass digital filters. Journal of Physical Oeanography, 12 112-115 .
    #
    #############################
    # Version 1.0  (12/4/96)  Jeff List (jlist@usgs.gov)
    # Version 1.1  (1/8/97)  Rich Signell (rsignell@usgs.gov)
    #     removed argument for number of points and add error trapping for matrices
    # Version 1.1b (12/1/2005) Rich Signell (rsignell@usgs.gov)
    #     added reference
    # (3/18/2019)
    # Translated to Python by André Palóczy.
    #############################
    npts = x.size
    if npts%2==0: # N is even.
        nby2 = npts//2
    else:
        nby2 = (npts-1)//2

    tfund = npts*dts
    ffund = 1.0/tfund

    xmean = x.mean()
    x -= xmean # Remove the mean from data.
    coeffs = np.fft.fft(x) # Fourier-transform data.

    # Filter coefficients.
    f = ffund
    for i in range(0, nby2+2):
        t = 1.0/f
        if np.logical_or(t>Tmax, t<Tmin):
            coeffs[i] = coeffs[i]*0
        f += ffund

    # Calculate the remaining coefficients.
    for i in range(1, nby2+2):
        coeffs[npts-i] = coeffs[i].conj()

    # Back-transform data and take real part.
    xfilt = np.fft.ifft(coeffs).real
    xfilt += xmean # Add back the mean.

    return xfilt


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


def near(x, x0, npts=1, return_index=False):
    """
    USAGE
    -----
    xnear = near(x, x0, npts=1, return_index=False)

    Finds 'npts' points (defaults to 1) in array 'x'
    that are closest to a specified 'x0' point.
    If 'return_index' is True (defauts to False),
    then the indices of the closest points are
    returned. The indices are ordered in order of
    closeness.
    """
    x = list(x)
    xnear = []
    xidxs = []
    for n in range(npts):
        idx = np.nanargmin(np.abs(np.array(x)-x0))
        xnear.append(x.pop(idx))
        if return_index:
            xidxs.append(idx)
    if return_index: # Sort indices according to the proximity of wanted points.
        xidxs = [xidxs[i] for i in np.argsort(xnear).tolist()]
    xnear.sort()

    if npts==1:
        xnear = xnear[0]
        if return_index:
            xidxs = xidxs[0]
    else:
        xnear = np.array(xnear)

    if return_index:
        return xidxs
    else:
        return xnear


def nearfl(x, x0):
    """
    Get the lowest index of the two points in
    vector x that bound the number x0.
    """
    return np.min(near(x, x0, npts=2, return_index=True))


d2r = np.pi/180
def sind(ang):
    return np.sin(ang*d2r)


def cosd(ang):
    return np.cos(ang*d2r)

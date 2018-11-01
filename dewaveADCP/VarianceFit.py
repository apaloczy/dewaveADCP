import numpy as np
from .utils import sind, cosd

def dewave_variancefit():
    # lstsq fit to find c1 and c2. Maybe Taylor-expand the cosh(c2*(z+h))
    zhp = z + h
    q = cosd(hdg)**2*sind(theta)**2
    bwave = c1*((q + Cth**2)*np.cosh(c2*zhp) + q - cosd(theta)**2)

    raise NotImplementedError

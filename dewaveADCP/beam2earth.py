# Functions for converting ADCP velocities in beam coordinates to instrument- or Earth-coordinates.
# Direct translation of functions in the 'ADCPtools' MATLAB
# package (https://github.com/apaloczy/ADCPtools).
import numpy as np
from utils import sind, cosd

d2r = np.pi/180


######################
#### 4-beam Janus ####
######################
def janus2xyz(b1, b2, b3, b4, theta, verbose=True):
    """
    USAGE
    -----
    vx, vy, vz = janus2xyz(b1, b2, b3, b4, theta)
    """
    Nz, Nt = b1.shape
    b1, b2 = b1[..., np.newaxis], b2[..., np.newaxis]
    b3, b4 = b3[..., np.newaxis], b4[..., np.newaxis]
    B = np.dstack((b1, b2, b3, b4))
    uvfac = 1/(2*sind(theta))
    wfac = 1/(4*cosd(theta)) # For w derived from beams 1-4.

    # 3rd row: w from the average of the 4 Janus beams.
    #               b1  b2  b3  b4
    A = np.array([[-1,  1,  0,  0],
                  [ 0,  0, -1,  1],
                  [-1, -1, -1, -1]])

    vxyz = np.zeros((Nz, Nt, 3))
    for nz in range(Nz):
        if verbose:
            print('Calculating Vx, Vy, Vz at bin ', nz, '/', Nz)
        for nt in range(Nt):
            vxyz[nz, nt, :] = np.matmul(A, B[nz, nt, :].T)

    Vx = vxyz[:, :, 0]*uvfac
    Vy = vxyz[:, :, 1]*uvfac
    Vz = vxyz[:, :, 2]*wfac

    return Vx, Vy, Vz


def janus2earth():
    raise NotImplementedError


######################
#### 5-beam Janus ####
######################
def janus2xyz5():
    raise NotImplementedError

def janus2earth5():
    raise NotImplementedError

# Functions for converting ADCP velocities in beam coordinates to instrument- or Earth-coordinates.
# Direct translation of functions in the 'ADCPtools' MATLAB
# package (https://github.com/apaloczy/ADCPtools).
import numpy as np
from .utils import sind, cosd

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
    if use3BeamSol:
        b1, b2, b3, b4 = janus3beamsol(b1, b2, b3, b4)

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

    vxyz = np.empty((Nz, Nt, 3))*np.nan
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
def janus2xyz5(b1, b2, b3, b4, b5, theta):
    """
    USAGE
    -----
    vx, vy, vz = janus2xyz5(b1, b2, b3, b4, b5, theta)
    """
    Nz, Nt = b1.shape

    b1, b2 = b1[..., np.newaxis], b2[..., np.newaxis]
    b3, b4 = b3[..., np.newaxis], b4[..., np.newaxis]
    b5 = b5[..., np.newaxis]
    B = np.dstack((b1, b2, b3, b4, b5))
    uvfac = 1/(2*sind(theta))
    wfac = 1/(4*cosd(theta)) # For w derived from beams 1-4.

    # 3rd row: w from the average of the 4 Janus beams.
    #               b1  b2  b3  b4
    A = np.array([[-1,  1,  0,  0,  0],
                  [ 0,  0, -1,  1,  0],
                  [-1, -1, -1, -1,  0],
                  [ 0,  0,  0,  0, -1]])

    vxyz = np.empty((Nz, Nt, 4))*np.nan
    for nz in range(Nz):
        if verbose:
            print('Calculating Vx, Vy, Vz at bin ', nz, '/', Nz)
        for nt in range(Nt):
            vxyz[nz, nt, :] = np.matmul(A, B[nz, nt, :].T)

    Vx = vxyz[:, :, 0]*uvfac
    Vy = vxyz[:, :, 1]*uvfac
    Vz = vxyz[:, :, 2]*wfac
    Vz5 = vxyz[:, :, 3]

    return Vx, Vy, Vz, Vz5





def janus2earth5():
    raise NotImplementedError


def binmap5(b1, b2, b3, b4, b5, r, r5, theta, ptch, roll, how='linear'):
    """
    USAGE
    -----

    b1m, b2m, b3m, b4m, b5m = binmap(b1, b2, b3, b4, b5, r, r5, theta, ptch, roll, how='linear')
    """
    return b1m, b2m, b3m, b4m, b5m


def janus3beamsol(b1, b2, b3, b4):
    """
    Usage
    -----
    b1, b2, b3, b4 = janus3beamsol(b1, b2, b3, b4)

    Calculates a three-beam solution for a bad beam when the other three Janus beams have good data.
    """
    [Nz Nt] = size(b1);

    for nt in range(Nt):
        for nz in range(Nz): # Set error velocity to zero: const*(b1 + b2 -b3 -b4) = 0.
            bki = np.array([b1[nz,nt] b2[nz,nt] b3[nz,nt] b4[nz,nt]])
            fbad = np.isnan(bki) # b1 + b2 = b3 + b4 Solve for bad beam.
            if fbad.sum()==1:    # Only one bad beam allowed for 3-beam solutions.
                fbad = np.where(fbad)[0][0]
            if fbad==1:   # Beam 1 is bad.
                b1[nz,nt] = bki[3] + bki[4] - bki[2]
            elif fbad==2: # Beam 2 is bad.
                b2[nz,nt] = bki[3] + bki[4] - bki[1]
            elif fbad==3: # Beam 3 is bad.
                b3[nz,nt] = bki[1] + bki[2] - bki[4]
            elif fbad==4: # Beam 4 is bad.
                b4[nz,nt] = bki[1] + bki[2] - bki[3]

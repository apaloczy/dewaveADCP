# Functions for converting ADCP velocities in beam coordinates to instrument- or Earth-coordinates.
# Direct translation of functions in the 'ADCPtools' MATLAB
# package (https://github.com/apaloczy/ADCPtools).
import numpy as np
from scipy.interpolate import interp1d
from .utils import sind, cosd, near, nearfl


######################
#### 4-beam Janus ####
######################
def janus2xyz(b1, b2, b3, b4, theta, r=None, ptch=None, roll=None, binmaptype=None, use3beamsol=True, verbose=True):
    """
    USAGE
    -----
    vx, vy, vz = janus2xyz(b1, b2, b3, b4, theta, r=None, ptch=None, roll=None, binmaptype=None, use3beamsol=True, verbose=True)

    theta, ptch, roll must be in RADIANS.
    """
    Nz, Nt = b1.shape
    if binmaptype is not None:
        assert r is not None, "Must provide r if using bin-mapping."
        assert ptch is not None, "Must provide pitch if using bin-mapping."
        assert roll is not None, "Must provide roll if using bin-mapping."
        print('Mapping bins to horizontal planes using *%s* interpolation.'%binmaptype)
        b1, b2, b3, b4 = binmap(b1, b2, b3, b4, r, theta, ptch, roll, how=binmaptype)
    else:
        print('Bin-mapping NOT applied.')

    if use3beamsol:
        b1, b2, b3, b4 = janus3beamsol(b1, b2, b3, b4)

    b1, b2 = b1[..., np.newaxis], b2[..., np.newaxis]
    b3, b4 = b3[..., np.newaxis], b4[..., np.newaxis]
    B = np.dstack((b1, b2, b3, b4))
    uvfac = 1/(2*np.sin(theta))
    wfac = 1/(4*np.cos(theta)) # For w derived from beams 1-4.

    # 3rd row: w from the average of the 4 Janus beams.
    #               b1  b2  b3  b4
    A = np.array([[-1,  1,  0,  0],
                  [ 0,  0, -1,  1],
                  [-1, -1, -1, -1]])

    vxyz = np.empty((Nz, Nt, 3))*np.nan
    for nz in range(Nz):
        if verbose:
            print('Calculating Vx, Vy, Vz at bin ', nz+1, '/', Nz)
        for nt in range(Nt):
            vxyz[nz, nt, :] = np.matmul(A, B[nz, nt, :].T)

    Vx = vxyz[:, :, 0]*uvfac
    Vy = vxyz[:, :, 1]*uvfac
    Vz = vxyz[:, :, 2]*wfac

    return Vx, Vy, Vz


def janus2earth(head, ptch, roll, theta, b1, b2, b3, b4, r=None, gimbaled=True, binmaptype=None, use3beamsol=True, verbose=True):
    """
     USAGE
     -----
     [u, v, w] = janus2earth(head, ptch, roll, theta, b1, b2, b3, b4, r=None, gimbaled=True, binmaptype=None, use3beamsol=True, verbose=True)

     Calculates Earth velocities (u,v,w) = (east,north,up) from beam-referenced velocity time series
     from a 4-beam Janus ADCP, (e.g., Appendix A of Dewey & Stringer (2007), Equations A3-A11).

     nz, nt, nb = number of vertical bins, data records, beams.

    ============================================================================
     For TRDI instruments, call function like this:
     u, v, w = janus2earth(head, ptch, roll, theta, b1, b2, b3, b4)

     For Nortek instruments, call function like this:
     u, v, w = janus2earth(head-90, roll, -ptch, theta, -b1, -b3, -b4, -b2)
    ============================================================================

        TRDI CONVENTION:
        ================

     * Velocity toward transducers' faces: POSITIVE
     * Clockwise PITCH (tilt about x-AXIS): POSITIVE (beam 3 higher than beam 4)
     * Clockwise ROLL (tilt about y-AXIS):  POSITIVE (beam 2 higher than beam 1)

     * Heading increases CLOCKWISE from the *Y-AXIS*.

           ^ positive y axis, psi = 0
           |
           3
           |
           |
           |
     2 --- O --- 1 ---> positive x axis, psi = +90
           |
           |
           |
           4

        NORTEK CONVENTION:
        ==================

     * Velocity toward transducers' faces: NEGATIVE
     * Counter-clockwise PITCH (tilt about y-AXIS, equivalent to -ROLL in the TRDI convention): POSITIVE (beam 1 higher than beam 3)
     * Clockwise ROLL (tilt about x-AXIS, equivalent to PITCH in the TRDI convention):  POSITIVE (beam 4 higher than beam 2)

     Heading increases CLOCKWISE from the *X-AXIS*.

           ^ positive y axis, psi = -90
           |
           4
           |
           |
           |
     3 --- O --- 1 ---> positive x axis, psi = 0
           |
           |
           |
           2

     INPUTS
     ------
     b1, b2, b3, b4        [nz -by- nt] matrices of along-beam velocity components.
     head, ptch, roll      [nt]         vectors with (time-dependent) heading, pitch
                                        and roll angles, following D&S2007's notation.

     theta                              Beam angle measured from the vertical.
                                        *For RDI Sentinel V and Nortek Signature: 25.

     gimbaled     [True or False]    Whether the ADCP was deployed with a gimbaled roll sensor
                                     (default true). Applies the correction to the raw pitch angle
                                     if the pitch/roll sensors were mounted rigidly to the
                                     instrument ('Gimbaled'==false), or the correction to the raw
                                     heading angle if the ADCP was mounted on a gimbal (Dewey &
                                     Stringer, 2007; Lohrmann et al., 1990).

     binmaptype [None or 'linear' or 'nn']
                                      Whether to map the beam velocities to fixed horizontal
                                      planes with linear interpolation ('linear') or nearest-neighbor
                                      interpolation ('nearest') prior to converting
                                      to instrument coordinates (Ott, 2002; Dewey & Stringer, 2007).
                                      *The default is to NOT perform any bin mapping.

     use3beamsol [True or False]      Whether to use three-beam solutions when exactly one beam has
                                      no data in one cell.

     OUTPUTS
     -------
     [u, v, w]           [east, north, up] components of Earth-referenced velocity vector.
    """
    nz, nt = b1.shape # Number of vertical bins and records in the time series.

    d2r = np.pi/180
    head = head*d2r
    ptch = ptch*d2r
    roll = roll*d2r
    theta = theta*d2r

    # Time-dependent angles (heading, pitch and roll).
    Sph1 = np.sin(head)
    Sph2 = np.sin(ptch)
    Sph3 = np.sin(roll)
    Cph1 = np.cos(head)
    Cph2 = np.cos(ptch)
    Cph3 = np.cos(roll)

    if gimbaled: # Correct heading (D&S 2007, eq. A2).
        print('Gimbaled instrument case.')
        Sph2Sph3 = Sph2*Sph3
        head = head + np.arcsin( Sph2Sph3/np.sqrt(Cph2**2 + Sph2Sph3**2) )
        Sph1 = np.sin(head)
        Cph1 = np.cos(head)
    else:                      # Correct pitch (D&S 2007, eq. A1 Lohrmann et al. 1990, eq. A1).
        print('Fixed instrument case.')
        ptch = np.arcsin( (Sph2*Cph3)/np.sqrt(1 - (Sph2*Sph3)**2) )
        Sph2 = np.sin(ptch)
        Cph2 = np.cos(ptch)

    # Convert instrument-referenced velocities
    # to Earth-referenced velocities.
    cx1 = Cph1*Cph3 + Sph1*Sph2*Sph3
    cx2 = Sph1*Cph3 - Cph1*Sph2*Sph3
    cx3 = Cph2*Sph3
    cy1 = Sph1*Cph2
    cy2 = Cph1*Cph2
    cy3 = Sph2
    cz1 = Cph1*Sph3 - Sph1*Sph2*Cph3
    cz2 = Sph1*Sph3 + Cph1*Sph2*Cph3
    cz3 = Cph2*Cph3

    # Convert beam-referenced velocities to instrument-referenced velocities.
    # NOTE: The convention used here (positive x axis = horizontally away from beam 1) and
    #                                 positive y axis = horizontally away from beam 3) is not
    #                                 the same as the one used by the instrument's firmware if
    #                                 the coordinate transformation mode is set to "instrument
    #                                 coordinates" before deployment.
    Vx, Vy, Vz = janus2xyz(b1, b2, b3, b4, theta, r=r, ptch=ptch, roll=roll, binmaptype=binmaptype, use3beamsol=use3beamsol, verbose=verbose)

    u = +Vx*cx1 + Vy*cy1 + Vz*cz1
    v = -Vx*cx2 + Vy*cy2 - Vz*cz2
    w = -Vx*cx3 + Vy*cy3 + Vz*cz3

    return u, v, w


######################
#### 5-beam Janus ####
######################
def janus2xyz5(b1, b2, b3, b4, b5, theta, r=None, ptch=None, roll=None, binmaptype=None, use3beamsol=True, verbose=True):
    """
    USAGE
    -----
    vx, vy, vz = janus2xyz5(b1, b2, b3, b4, b5, theta, r=None, ptch=ptch, roll=roll, binmaptype=None, use3beamsol=True, verbose=True)

    theta, ptch, roll must be in RADIANS.
    """
    Nz, Nt = b1.shape
    if binmaptype is not None:
        assert r is not None, "Must provide r if using bin-mapping."
        assert ptch is not None, "Must provide pitch if using bin-mapping."
        assert roll is not None, "Must provide roll if using bin-mapping."
        print('Mapping bins to horizontal planes using *%s* interpolation.'%binmaptype)
        b1, b2, b3, b4, b5 = binmap5(b1, b2, b3, b4, b5, r, theta, ptch, roll, how=binmaptype)
    else:
        print('Bin-mapping NOT applied.')

    if use3beamsol:
        b1, b2, b3, b4 = janus3beamsol(b1, b2, b3, b4)

    b1, b2 = b1[..., np.newaxis], b2[..., np.newaxis]
    b3, b4 = b3[..., np.newaxis], b4[..., np.newaxis]
    b5 = b5[..., np.newaxis]
    B = np.dstack((b1, b2, b3, b4, b5))
    uvfac = 1/(2*np.sin(theta))
    wfac = 1/(4*np.cos(theta)) # For w derived from beams 1-4.

    # 3rd row: w from the average of the 4 Janus beams.
    #               b1  b2  b3  b4  b5
    A = np.array([[-1,  1,  0,  0,  0],
                  [ 0,  0, -1,  1,  0],
                  [-1, -1, -1, -1,  0],
                  [ 0,  0,  0,  0, -1]])

    vxyz = np.empty((Nz, Nt, 4))*np.nan
    for nz in range(Nz):
        if verbose:
            print('Calculating Vx, Vy, Vz, Vz5 at bin ', nz+1, '/', Nz)
        for nt in range(Nt):
            vxyz[nz, nt, :] = np.matmul(A, B[nz, nt, :].T)

    Vx = vxyz[:, :, 0]*uvfac
    Vy = vxyz[:, :, 1]*uvfac
    Vz = vxyz[:, :, 2]*wfac
    Vz5 = vxyz[:, :, 3]

    return Vx, Vy, Vz, Vz5


def janus2earth5(head, ptch, roll, theta, b1, b2, b3, b4, b5, r=None, gimbaled=True, binmaptype=None, uvwbeam5=True, use3beamsol=True, verbose=True):
    """
     USAGE
     -----
     u, v, w, w5 = janus2earth5(head, ptch, roll, theta, b1, b2, b3, b4, b5, r=None, gimbaled=True, binmaptype=None, uvwbeam5=True, use3beamsol=True, verbose=True)

     Calculates Earth velocities (u,v,w) = (east,north,up) from beam-referenced velocity time series
     from a 5-beam Janus ADCP, (e.g., Appendix A of Dewey & Stringer (2007), Equations A3-A11).

     nz, nt, nb = number of vertical bins, data records, beams.

    ============================================================================
     For TRDI instruments, call function like this:
     u, v, w = janus2earth(head, ptch, roll, theta, b1, b2, b3, b4)

     For Nortek instruments, call function like this:
     u, v, w = janus2earth(head-90, roll, -ptch, theta, -b1, -b3, -b4, -b2)
    ============================================================================

        TRDI CONVENTION:
        ================

     * Velocity toward transducers' faces: POSITIVE
     * Clockwise PITCH (tilt about x-AXIS): POSITIVE (beam 3 higher than beam 4)
     * Clockwise ROLL (tilt about y-AXIS):  POSITIVE (beam 2 higher than beam 1)

     * Heading increases CLOCKWISE from the *Y-AXIS*.

           ^ positive y axis, psi = 0
           |
           3
           |
           |
           |
     2 --- O --- 1 ---> positive x axis, psi = +90
           |
           |
           |
           4

        NORTEK CONVENTION:
        ==================

     * Velocity toward transducers' faces: NEGATIVE
     * Counter-clockwise PITCH (tilt about y-AXIS, equivalent to -ROLL in the TRDI convention): POSITIVE (beam 1 higher than beam 3)
     * Clockwise ROLL (tilt about x-AXIS, equivalent to PITCH in the TRDI convention):  POSITIVE (beam 4 higher than beam 2)

     Heading increases CLOCKWISE from the *X-AXIS*.

           ^ positive y axis, psi = -90
           |
           4
           |
           |
           |
     3 --- O --- 1 ---> positive x axis, psi = 0
           |
           |
           |
           2

     INPUTS
     ------
     b1, b2, b3, b4, b5    [nz -by- nt] matrices of along-beam velocity components.
     head, ptch, roll      [nt]         vectors with (time-dependent) heading, pitch
                                        and roll angles, following D&S2007's notation.

     theta                              Beam angle measured from the vertical.
                                        *For RDI Sentinel V and Nortek Signature: 25.

     uvwBeam5     [True or False]    whether to calculate [u, v, w] using the independent information
                                     from beam 5 (defaults true). If false, the usual four-beam
                                     solution using w derived from beams 1-4 is calculated.

     gimbaled     [True or False]    Whether the ADCP was deployed with a gimbaled roll sensor
                                     (default true). Applies the correction to the raw pitch angle
                                     if the pitch/roll sensors were mounted rigidly to the
                                     instrument ('Gimbaled'==false), or the correction to the raw
                                     heading angle if the ADCP was mounted on a gimbal (Dewey &
                                     Stringer, 2007; Lohrmann et al., 1990).

     binmaptype [None or 'linear' or 'nn']
                                      Whether to map the beam velocities to fixed horizontal
                                      planes with linear interpolation ('linear') or nearest-neighbor
                                      interpolation ('nearest') prior to converting
                                      to instrument coordinates (Ott, 2002; Dewey & Stringer, 2007).
                                      *The default is to NOT perform any bin mapping.

     use3beamsol [True or False]      Whether to use three-beam solutions when exactly one beam has
                                      no data in one cell.

     OUTPUTS
     -------
     [u, v, w, w5]           [east, north, up, up-(from vertical beam only)] components
                              of Earth-referenced velocity vector.
    """
    nz, nt = b1.shape # Number of vertical bins and records in the time series.

    d2r = np.pi/180
    head = head*d2r
    ptch = ptch*d2r
    roll = roll*d2r
    theta = theta*d2r

    # Time-dependent angles (heading, pitch and roll).
    Sph1 = np.sin(head)
    Sph2 = np.sin(ptch)
    Sph3 = np.sin(roll)
    Cph1 = np.cos(head)
    Cph2 = np.cos(ptch)
    Cph3 = np.cos(roll)

    if gimbaled: # Correct heading (D&S 2007, eq. A2).
        print('Gimbaled instrument case.')
        Sph2Sph3 = Sph2*Sph3
        head = head + np.arcsin( Sph2Sph3/np.sqrt(Cph2**2 + Sph2Sph3**2) )
        Sph1 = np.sin(head)
        Cph1 = np.cos(head)
    else:                      # Correct pitch (D&S 2007, eq. A1 Lohrmann et al. 1990, eq. A1).
        print('Fixed instrument case.')
        ptch = np.arcsin( (Sph2*Cph3)/np.sqrt(1 - (Sph2*Sph3)**2) )
        Sph2 = np.sin(ptch)
        Cph2 = np.cos(ptch)

    # Convert instrument-referenced velocities
    # to Earth-referenced velocities.
    # Option 1: Classic four-beam solution.
    # Option 2: five-beam solution for [u, v, w].
    cx1 = Cph1*Cph3 + Sph1*Sph2*Sph3
    cx2 = Sph1*Cph3 - Cph1*Sph2*Sph3
    cx3 = Cph2*Sph3
    cy1 = Sph1*Cph2
    cy2 = Cph1*Cph2
    cy3 = Sph2
    cz1 = Cph1*Sph3 - Sph1*Sph2*Cph3
    cz2 = Sph1*Sph3 + Cph1*Sph2*Cph3
    cz3 = Cph2*Cph3

    # Convert beam-referenced velocities to instrument-referenced velocities.
    # NOTE: The convention used here (positive x axis = horizontally away from beam 1) and
    #                                 positive y axis = horizontally away from beam 3) is not
    #                                 the same as the one used by the instrument's firmware if
    #                                 the coordinate transformation mode is set to "instrument
    #                                 coordinates" before deployment.
    Vx, Vy, Vz, Vz5 = janus2xyz5(b1, b2, b3, b4, b5, theta, r=r, ptch=ptch, roll=roll, binmaptype=binmaptype, use3beamsol=use3beamsol, verbose=verbose)

    w5 = Vz5*cz3 # w from beam 5 only.

    if uvwbeam5:
        print('Using vertical beam for [u, v, w].')
        u = +Vx*cx1 + Vy*cy1 + Vz5*cz1
        v = -Vx*cx2 + Vy*cy2 - Vz5*cz2
        w = -Vx*cx3 + Vy*cy3 + w5
    else:
        print('Using only beams 1-4 for [u, v, w].')
        u = +Vx*cx1 + Vy*cy1 + Vz*cz1
        v = -Vx*cx2 + Vy*cy2 - Vz*cz2
        w = -Vx*cx3 + Vy*cy3 + Vz*cz3

    return u, v, w, w5



def binmap(b1, b2, b3, b4, r, theta, ptch, roll, how='linear'):
    """
    USAGE
    -----
    b1m, b2m, b3m, b4m = binmap(b1, b2, b3, b4, r, theta, ptch, roll, how='linear')

    theta, ptch and roll must be in RADIANS.

    Interpolate beam-coordinate velocities to fixed horizontal planes based on tilt angles
    (pitch and roll).
    """
    Sth = np.sin(theta)
    Cth = np.cos(theta)

    Sph2 = np.sin(ptch)
    Cph2 = np.cos(ptch)
    Sph3 = np.sin(roll)
    Cph3 = np.cos(roll)

    Z = r*Cth
    z00 = np.matrix([0, 0, -1]).T

    nz, nt = b1.shape
               #      b1     b2     b3     b4
    E = np.matrix([[-Sth,  +Sth,    0,     0],
                   [ 0,      0,   -Sth,  +Sth],
                   [-Cth,  -Cth,  -Cth,  -Cth]])

    Bo = np.dstack((b1[..., np.newaxis], b2[..., np.newaxis], b3[..., np.newaxis], b4[..., np.newaxis]))

    for i in range(4):
        Ei = E[:,i]

        Boi = Bo[:,:,i] # z, t, bi.
        bmi = Boi.copy()

        for k in range(nt):
            PR = np.array([[Cph3[k],             0,     Sph3[k]],
                           [Sph2[k]*Sph3[k],  Cph2[k], -Sph2[k]*Cph3[k]],
                           [-Sph3[k]*Cph2[k], Sph2[k],  Cph2[k]*Cph3[k]]])

            zi = np.array((PR*Ei).T*z00*r).squeeze() # Actual bin height, dot product of tilt matrix with along-beam distance vector.
            bmi[:,k] = interp1d(zi, Boi[:,k], kind=how, fill_value="extrapolate", assume_sorted=True)(Z)

        Bo[:,:,i] = bmi

    return Bo[:,:,0], Bo[:,:,1], Bo[:,:,2], Bo[:,:,3]


def binmap5(b1, b2, b3, b4, b5, r, theta, ptch, roll, how='linear'):
    """
    USAGE
    -----
    b1m, b2m, b3m, b4m, b5m = binmap5(b1, b2, b3, b4, b5, r, theta, ptch, roll, how='linear')

    theta, ptch and roll must be in RADIANS.

    Interpolate beam-coordinate velocities to fixed horizontal planes based on tilt angles
    (pitch and roll).
    """
    Sth = np.sin(theta)
    Cth = np.cos(theta)

    Sph2 = np.sin(ptch)
    Cph2 = np.cos(ptch)
    Sph3 = np.sin(roll)
    Cph3 = np.cos(roll)

    Z = r*Cth
    z00 = np.matrix([0, 0, -1]).T

    nz, nt = b1.shape
               #      b1     b2     b3     b4   b5
    E = np.matrix([[-Sth,  +Sth,    0,     0,   0],
                   [ 0,      0,   -Sth,  +Sth,  0],
                   [-Cth,  -Cth,  -Cth,  -Cth, -1]])

    Bo = np.dstack((b1[..., np.newaxis], b2[..., np.newaxis], b3[..., np.newaxis], b4[..., np.newaxis], b5[..., np.newaxis]))

    for i in range(5):
        Ei = E[:,i]

        Boi = Bo[:,:,i] # z, t, bi.
        bmi = Boi.copy()

        for k in range(nt):
            PR = np.array([[Cph3[k],             0,     Sph3[k]],
                            [Sph2[k]*Sph3[k],  Cph2[k], -Sph2[k]*Cph3[k]],
                            [-Sph3[k]*Cph2[k], Sph2[k],  Cph2[k]*Cph3[k]]])

            zi = np.array((PR*Ei).T*z00*r).squeeze() # Actual bin height, dot product of tilt matrix with along-beam distance vector.
            bmi[:,k] = interp1d(zi, Boi[:,k], kind=how, fill_value="extrapolate", assume_sorted=True)(Z)

        Bo[:,:,i] = bmi

    return Bo[:,:,0], Bo[:,:,1], Bo[:,:,2], Bo[:,:,3], Bo[:,:,4]


def janus3beamsol(b1, b2, b3, b4):
    """
    Usage
    -----
    b1, b2, b3, b4 = janus3beamsol(b1, b2, b3, b4)

    Calculates a three-beam solution for a bad beam when the other three Janus beams have good data.
    """
    Nz, Nt = b1.shape

    for nt in range(Nt):
        for nz in range(Nz): # Set error velocity to zero: const*(b1 + b2 -b3 -b4) = 0.
            bki = np.array([b1[nz,nt], b2[nz,nt], b3[nz,nt], b4[nz,nt]])
            fbad = np.isnan(bki) # b1 + b2 = b3 + b4 Solve for bad beam.
            if fbad.sum()==1:    # Only one bad beam allowed for 3-beam solutions.
                fbad = np.where(fbad)[0][0]
                if fbad==0:   # Beam 1 is bad.
                    b1[nz,nt] = bki[2] + bki[3] - bki[1]
                elif fbad==1: # Beam 2 is bad.
                    b2[nz,nt] = bki[2] + bki[3] - bki[0]
                elif fbad==2: # Beam 3 is bad.
                    b3[nz,nt] = bki[0] + bki[1] - bki[3]
                elif fbad==3: # Beam 4 is bad.
                    b4[nz,nt] = bki[0] + bki[1] - bki[2]

    return b1, b2, b3, b4

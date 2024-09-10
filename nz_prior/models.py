import numpy as np
from scipy.interpolate import interp1d

def shift_model(nz, shift):
    """
    Aplies a shift to the given p(z) distribution.
    This is done by interpolating the p(z) distribution
    at the shifted z values and then evaluating it at the
    original z values.
    """
    z = nz[0]
    nz = nz[1]
    nz_i = interp1d(z, nz,
                    kind='linear',
                    fill_value='extrapolate')
    pdf = nz_i(z+shift)
    norm = np.sum(pdf)
    return [z, pdf/norm]

def shift_and_width_model(nz, shift, width):
    """
    Aplies a shift and a width to the given p(z) distribution.
    This is done by evluating the n(z) distribution at
    p((z-mu)/width + mu + shift) where mu is the mean redshift
    of the fiducial n(z) distribution and the rescaling by the width.
    Finally the distribution is normalized.
    """
    z = nz[0]
    nz = nz[1]
    nz_i = interp1d(z, nz, kind='linear', fill_value='extrapolate')
    mu = np.average(z, weights=nz)
    pdf = nz_i((z-mu)/width + mu + shift)/width
    norm = np.sum(pdf)
    return [z, pdf/norm]

def comb_model(nz, W, combs):
    M = len(W)
    z = nz[0]
    nz = nz[1]
    dz = (np.max(z) - np.min(z))/M 
    zmeans = [(np.min(z)+dz/2) + i*dz for i in range(M)]
    combs = {}
    for i in np.arange(M):
        combs[i] =  stats.norm(zmeans[i], dz/2)
    nz_pred = np.zeros(len(z))
    for i in np.arange(M):
        nz_pred += (W[i]/M)*combs[i].pdf(z)
    norm = np.sum(nz_pred)
    return [z, nz_pred/norm]

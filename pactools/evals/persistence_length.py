#!/bin/env python3

import sys
try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit(1)

from pactools.tools import conditional_numba as condnumba


########################################################################
########################################################################
########################################################################

def cal_persistence_length(conf: np.ndarray ,m_max=1,disc_len=None):
    """ calculates the persistence length for the given set of configurations """
    if len(np.shape(conf)) == 2:
        conf = np.array([conf])
    if disc_len is None:
        disc_len = cal_disc_len(conf)

    tans =  __get_tangents(conf,normalized=True)
    if m_max == 1:
        lb = __cal_perslen_single(tans,disc_len)
    else:
        lb = __cal_perslen_multiple(tans, m_max, disc_len)
    return lb

@condnumba
def __cal_perslen_single(tans: np.ndarray,disc_len: float) -> float:
    tancor = 0.
    for s in range(len(tans)):
        for i in range(len(tans[0])-1):
            tancor += np.dot(tans[s,i],tans[s,i+1])

    tancor /= len(tans)*(len(tans[0])-1)
    return -disc_len/np.log(tancor)


@condnumba
def __cal_perslen_multiple(tans: np.ndarray, m_max: int, disc_len: float) -> np.ndarray:
    tancor = np.zeros(m_max)
    numcor = np.zeros(m_max)
    for s in range(len(tans)):
        for i in range(len(tans[0])-1):
            for m in range(m_max):
                j = i+1+m
                if j >= len(tans[0]):
                    break
                tancor[m] += np.dot(tans[s,i],tans[s,i+1+m])
                numcor[m] += 1
    return -disc_len*np.arange(1,m_max+1) / np.log(tancor/numcor)


def cal_disc_len(conf: np.ndarray) -> float:
    """ returns the mean discretization lenght. """
    if len(np.shape(conf)) == 2:
        conf = np.array([conf])
    return np.round(__cal_disc_len(conf), decimals=8)

@condnumba
def __cal_disc_len(conf: np.ndarray) -> float:
    dlen = 0.0
    for s in range(len(conf)):
        for i in range(len(conf[0])-1):
            dlen += np.linalg.norm(conf[s,i+1]-conf[s,i])
    return dlen/(len(conf)*(len(conf[0])-1))


def get_tangents(conf: np.ndarray, normalized=False) -> np.ndarray:
    """ returns tangents for given configuration. Return matrix will be of dim (m,n,3) """
    if len(np.shape(conf)) == 2:
        conf = np.array([conf])
    return __get_tangents(conf,normalized=normalized)

@condnumba
def __get_tangents(conf: np.ndarray,normalized=False) -> np.ndarray:
    tans = np.zeros((len(conf), len(conf[0])-1, len(conf[0,0])))
    for s in range(len(conf)):
        for i in range(len(conf[0])-1):
            tans[s,i] = conf[s, i + 1] - conf[s, i]
            if normalized:
                tans[s, i] /= np.linalg.norm(tans[s,i])
    return tans



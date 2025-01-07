#!/bin/env python3

import sys
try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit(1)

########################################################################
########################################################################
########################################################################

AVOGADROS_NUMBER = 6.02214076*10**23

def numdens2N(c: float, L: float) -> int:
    return int(np.ceil(c*L**3))

def mol2N(c: float, L: float):
    V = L**3 * 1000
    return int(np.ceil(c * AVOGADROS_NUMBER * V))

def mumol2N_in_nm(c_muM: float, L: float):
    V = boxL2lites_in_nm(L)
    ctot = c_muM*10**-6
    return int(np.ceil(ctot * AVOGADROS_NUMBER * V))

def mumol2numdens_in_nm(c_muM):
    return c_muM*10**-6 * AVOGADROS_NUMBER * (10**-9)**3 * 1000

def boxL2lites_in_nm(L: float):
    return (L*10**-9)**3 * 1000


if __name__ == "__main__":

    L     = float(sys.argv[1])
    c_muM = float(sys.argv[2])

    N = mumol2N_in_nm(L, c_muM)
    print(f'N = {N}')
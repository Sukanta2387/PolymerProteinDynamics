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
### Radius of Gyration #################################################

# @jit(nopython=True)
def radius_of_gyration(conf: np.ndarray):
    if len(np.shape(conf)) not in [2,3]:
        raise ValueError(f"Dimension of configuration matrix needs to be 2 or 3. {len(np.shape(conf))} given.")
    if len(np.shape(conf)) == 3:
        rgs = np.zeros(len(conf))
        for i in range(len(rgs)):
            rgs[i] = __radius_of_gyration(conf[i])
        return rgs
    return __radius_of_gyration(conf)

def __radius_of_gyration(conf):
    com  = center_of_mass(conf)
    return np.sqrt(np.sum((conf-com)**2))

def center_of_mass(conf):
    return np.mean(conf,axis=0)

########################################################################
########################################################################
### Radial Distribution Function #######################################

# def radial_distribution_function(confs):





if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s num_bp disc_len" % sys.argv[0])
        sys.exit()

    nbp = int(sys.argv[1])


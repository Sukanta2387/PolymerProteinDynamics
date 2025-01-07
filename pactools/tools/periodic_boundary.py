#!/bin/env python3

import sys
from pactools.tools.conditional_numba import conditional_numba as condnumba

try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit("terminated")

########################################################################
########################################################################
########################################################################
# impose periodic boundary for single atom

@condnumba
def place_in_box_single(box: np.ndarray, r: np.ndarray)-> np.ndarray:
    rnew = np.zeros(np.shape(r))
    rnew[0] = (r[0]-box[0,0])%(box[0,1]-box[0,0]) + box[0,0]
    rnew[1] = (r[1]-box[1,0])%(box[1,1]-box[1,0]) + box[1,0]
    rnew[2] = (r[2]-box[2,0])%(box[2,1]-box[2,0]) + box[2,0]
    return rnew

########################################################################
# impose periodic boundary for array of atoms

@condnumba
def place_in_box(box: np.ndarray, R: np.ndarray) -> np.ndarray:
    Rnew = np.zeros(np.shape(R))
    span = box[:,1] - box[:,0]
    for i in range(len(R)):
        Rnew[i,0] = (R[i,0]-box[0,0])%span[0] + box[0,0]
        Rnew[i,1] = (R[i,1]-box[1,0])%span[1] + box[1,0]
        Rnew[i,2] = (R[i,2]-box[2,0])%span[2] + box[2,0]
    return Rnew

def valid_box_dimension(box: np.ndarray):
    if np.shape(box) != (3,2):
        raise ValueError(f'Invalid dimension of box. Needs to be (3x2), {np.shape(box)} given')

def valid_box(box: np.ndarray):
    if np.shape(box) != (3,2):
        raise ValueError(f'Invalid dimension of box. Needs to be (3x2), {np.shape(box)} given')
    for i in range(3):
        if box[i,1] <= box[i,0]:
            raise ValueError(f'Lower bound of periodic box larger than upper bound')

# @condnumba
def is_outside_box(pos: np.ndarray,box: np.ndarray):
    for d in range(3):
        if pos[d] <= box[d,0]:
            return True
        if pos[d] >= box[d,1]:
            return True
    return False
        


########################################################################
# Unwrap Coordinates

def unwrap_polymer(conf: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    Unwraps the corrodinates of the provided chain. The configutation matrix can only contain the possitions
    DNA monomers
    :param conf:        positions of DNA monomers. The dimension may be (m,n,d) or (n,d), with m the number of snapshots
                        n the number of monomers and d the dimensionality of the space
    :param box:         limits of the periodic boxy
    :param disc_len (optional):
                        discretization length of the chain. If not provided the discretization length is calculated
                        based on the closed monomer distance found in the first snapshot
    :return:    matrix of the same dimension as conf with unwrapped coordinates
    """
    if len(np.shape(conf)) not in [2,3]:
        raise ValueError(f"Dimension of configuration matrix needs to be 2 or 3. {len(np.shape(conf))} given.")
    if len(np.shape(conf)) == 3:
        # uconfs = np.empty(np.shape(conf))
        # disc_len = unwrap_disc_len(conf[0])
        # for i in range(len(uconfs)):
        #     uconfs[i] = __unwrap_polymer(conf[i], box, disc_len)
        # return uconfs
        # disc_len = unwrap_disc_len(conf[0])
        return __loop_unwrap_dnp(conf,box)
    return __unwrap_polymer(conf, box)

@condnumba
def __loop_unwrap_dnp(conf: np.ndarray, box: np.ndarray) -> np.ndarray:
    uconfs = np.empty(np.shape(conf))
    for i in range(len(uconfs)):
        uconfs[i] = __unwrap_polymer(conf[i], box)
    return uconfs

@condnumba
def __unwrap_polymer(conf: np.ndarray, box: np.ndarray) -> np.ndarray:
    uconf = np.zeros(np.shape(conf))
    uconf[0] = conf[0]
    boxL = box[:,1] - box[:,0]
    dim   = np.shape(conf)[-1]
    for i in range(len(conf)-1):
        for d in range(dim):
            uconf[i+1,d] = conf[i+1,d]-np.round((conf[i+1,d]-uconf[i,d])/boxL[d])*boxL[d]
    return uconf


# @jit(nopython=True)
def unwrap_disc_len(conf: np.ndarray) -> np.ndarray:
    Ts = np.diff(conf,n=1,axis=0)
    return np.min(np.linalg.norm(Ts,axis=1))


########################################################################

def center_polymer(conf: np.ndarray, box: np.ndarray):
    unwrapped = unwrap_polymer(conf,box)
    com = np.mean(unwrapped,axis=0)
    center = 0.5 * (box[:,0] + box[:,1])
    return unwrapped - com + center


########################################################################
# script call only for testing purposes

if __name__ == "__main__":

    box = np.array([[-2.,10],[-2.,10],[-2,10]])
    # box = np.array([[-2.,10],[-2,10]])


    r = np.array([[5,80,-9],[-456,46.57,-2.001]])
    print(r)
    r = place_in_box(box,r)

    print(r)




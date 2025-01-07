#!/bin/env python3
import sys
from typing import List, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

# try:
#     from numba.experimental import jitclass
#     from numba import float64
#     use_numba = True
# except ModuleNotFoundError:
#     use_numba = False
from pactools.tools import conditional_jitclass
# from pactools.interactions import Pair

LJ_SIXROOTTWO = 1.122462048309373

def lj_min(sigma):
    return LJ_SIXROOTTWO*sigma

@conditional_jitclass
class LJ():

    rc:     float
    eps:    float
    sigma:  float
    shiftval: float

    def __init__(self ,eps: float ,sigma: float, rc: float):
        self.eps    = eps
        self.sigma  = sigma
        self.rc     = rc
        self.shiftval = self.eval(rc,shift=False)

    def eval(self,r: float,shift=True):
        if r > self.rc:
            return 0
        fac = (self.sigma/r)**6
        E = 4*self.eps*(fac**2-fac)
        if shift:
            E -= self.shiftval
        return E




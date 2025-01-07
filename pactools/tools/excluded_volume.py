#!/bin/env python3

import sys
from typing import List,Tuple

from pactools.tools.conditional_numba import conditional_numba as condnumba
from pactools.tools.periodic_boundary import is_outside_box

try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit("terminated")

########################################################################
########################################################################
########################################################################


def check_ev_violation_single(pos: np.ndarray, radius: float, existing_atoms: np.ndarray, boundary: str, box=None) -> bool:
    if len(existing_atoms[0]) < 4:
        raise ValueError(f"Entries of existing_atoms require 4 entries: The three dimensions and the radius of the atom.")

    R        = existing_atoms[:,:3]
    ev_dists = existing_atoms[:,3] + radius

    if boundary == 'periodic':
        return ev_violation_periodic_box_individual_ev_dist(R,pos,ev_dists, box)
    else:
        return ev_violation_individual_ev_dist(R, pos, ev_dists)

def conf2existing(conf: np.ndarray, radius: float) -> np.ndarray:
    existing_atoms = np.zeros((len(conf),4))
    existing_atoms[:,:3] = conf
    existing_atoms[:,3]  = radius
    return existing_atoms

def confs2existing(confs: List[np.ndarray], radii: List[float]) -> np.ndarray:
    if len(confs) == 0:
        return None
    num = np.sum([len(conf) for conf in confs])
    existing_atoms = np.zeros((num,4))
    cid = 0
    for i,conf in enumerate(confs):
        existing_atoms[cid:cid+len(conf),:3] = conf
        existing_atoms[cid:cid+len(conf),3]  = radii[i]
        cid += len(conf)
    return existing_atoms

def existing2conf(existing_atoms: np.ndarray) -> Tuple[np.ndarray]:
    return existing_atoms[:,:3],existing_atoms[:,3]

########################################################################
########################################################################
########################################################################

@condnumba
def ev_violation_pair(r1: np.ndarray,r2: np.ndarray, ev_dist: float) -> bool:
    """ checks if pair of atoms violates excluded volume set by ev_dist """
    if np.linalg.norm(r2-r1) < ev_dist:
        return True
    return False

@condnumba
def ev_violation_pair_relsize(r1: np.ndarray, r2: np.ndarray, size1: float,size2: float) -> bool:
    """ checks if pair of atoms violates excluded volume set by the respective size of the atoms """
    ev_dist = 0.5*(size1+size2)
    return ev_violation_pair(r1, r2, ev_dist)

@condnumba
def ev_violation(R: np.ndarray, ev_dist: float, excluded_neighbors=0) -> bool:
    """ checks for excluded volume violations within the given group of atoms """
    N = len(R)
    for i in range(excluded_neighbors+1,N):
        for j in range(0,i-excluded_neighbors):
            if np.linalg.norm(R[j] - R[i]) < ev_dist:
                return True
    return False

@condnumba
def find_ev_violation(R: np.ndarray, ev_dist: float, excluded_neighbors=0) -> int:
    """ checks for excluded volume violations within the given group of atoms """
    N = len(R)
    for i in range(excluded_neighbors+1,N):
        for j in range(0,i-excluded_neighbors):
            if np.linalg.norm(R[j] - R[i]) < ev_dist:
                return i
    return -1

@condnumba
def ev_violation_single(R: np.ndarray, r: np.ndarray, ev_dist: np.ndarray) -> bool:
    """ checks if the atom specified by r violates excluded volumes with the group of atoms R """
    for i in range(len(R)):
        if np.linalg.norm(r - R[i]) < ev_dist:
            return True
    return False

@condnumba
def ev_violation_individual_ev_dist(R: np.ndarray, r: np.ndarray, ev_dists: np.ndarray) -> bool:
    """ checks if the atom specified by r violates excluded volumes with the group of atoms R. For each atom
    individual excluded volume distances have to be specified """
    for i in range(len(R)):
        if np.linalg.norm(r - R[i]) < ev_dists[i]:
            return True
    return False

########################################################################
# including periodic boundary conditions

@condnumba
def ev_violation_single_periodic_box(R: np.ndarray,r: np.ndarray,ev_dist: float,periodic_box: np.ndarray) -> bool:
    """ checks if the atom specified by r violates excluded volumes with the group of atoms R including periodic boundary"""
    for i in range(len(R)):
        if ev_violation_pair_in_periodic_box(r,R[i],ev_dist,periodic_box):
            return True
    return False

@condnumba
def ev_violation_periodic_box_individual_ev_dist(R: np.ndarray,r: np.ndarray,ev_dists: np.ndarray, periodic_box: np.ndarray) -> bool:
    """ checks if the atom specified by r violates excluded volumes with the group of atoms R including periodic boundary. For each atom
    individual excluded volume distances have to be specified """
    # if len(ev_dists) != len(R):
    #     raise ValueError(f'The number of specified ev_dists needs to match the number of positions R. len(ev_dists) = {len(ev_dists)} and len(R) = {len(R)}.')
    for i in range(len(R)):
        if ev_violation_pair_in_periodic_box(r,R[i],ev_dists[i],periodic_box):
            return True
    return False

@condnumba
def ev_violation_pair_in_periodic_box(r1: np.ndarray,r2: np.ndarray,ev_dist: float,periodic_box: np.ndarray) -> bool:
    """
     checks if the two atoms or their periodic copies violate the excluded volume set by ev_dist
     assumes the tho positions to be within the box
    """
    dists = np.zeros(3)
    for i in range(3):
        dists[i] = closest_dist_1d_periodic(r1[i], r2[i], periodic_box[i])
    if np.linalg.norm(dists) < ev_dist:
        return True
    return False

@condnumba
def closest_dist_1d_periodic(a: float, b: float , box_bound: np.ndarray) -> float:
    """ calculates the closest distance of two points in 1d across the periodic boundary """
    dx = np.abs(a - b)
    box_range = box_bound[1] - box_bound[0]
    if a > box_bound[0] + box_range*0.5:
        dx_alt = np.abs(a - box_range - b)
    else:
        dx_alt = np.abs(a + box_range - b)
    if dx_alt < dx:
        return dx_alt
    return dx


########################################################################
# script call only for testing purposes

if __name__ == "__main__":

    periodic_box = np.array([[-2.,10],[-2.,10],[-2,10]])
    # box = np.array([[-2.,10],[-2,10]])

    r1 = np.array([0,0,0])
    r2 = np.array([0.5,0,0])
    ev_dist = 1

    for i in range(1000000):
        if i%100000==0:
            print(i)
        violation = ev_violation_pair_in_periodic_box(r1,r2,ev_dist,periodic_box)
    print(violation)




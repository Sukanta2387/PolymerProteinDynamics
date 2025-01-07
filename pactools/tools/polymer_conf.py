#!/bin/env python3

import sys
from typing import List, Tuple

from pactools.tools.SO3Methods import get_rot_mat
# from .excluded_volume import ev_violation_single, ev_violation_single_periodic_box, ev_violation_pair_in_periodic_box
from pactools.tools.excluded_volume import check_ev_violation_single, ev_violation_single, ev_violation_single_periodic_box, ev_violation_periodic_box_individual_ev_dist,ev_violation_individual_ev_dist
from pactools.tools.periodic_boundary import valid_box, place_in_box_single, place_in_box, unwrap_polymer, is_outside_box
from pactools.tools.conditional_numba import conditional_numba as condnumba
from pactools.tools.random_placement import gen_random_point_in_box

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy: pip install numpy")
    sys.exit("terminated")

#import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d

########################################################################
########################################################################
########################################################################

BOUNDARY_NONE       = 'none'
BOUNDARY_PERIODIC   = 'periodic'
BOUNDARY_REFLECTING = 'reflecting'
VALID_BOUNDARYS     = ['none', 'periodic', 'reflecting']

GENPOLYMER_MAXGENERATION_DEPTH = 10000
GENPOLYMER_MAXTRIALSPERSTEP    = 1000

########################################################################
########################################################################
########################################################################

def gen_polymer(
    num_monomer: int,
    disc_len: float,
    boundary=BOUNDARY_NONE,
    box=None,
    lb=40.0,
    radius=0.0,
    existing_atoms=None,
    first_pos=None,
    first_triad=None,
    ) -> np.ndarray:
    """
    Generates a DNA configuration. periodic boundaries will be imposed if the periodic_box argument is passed.
    Excluded columes between atoms will be imposed based on an excluded volume radius set by the argument.
    """

    # verify that provided boundary setting is valid
    boundary = boundary.lower()
    if boundary not in VALID_BOUNDARYS:
        valid_bsettings = ''
        for vbs in VALID_BOUNDARYS:
            valid_bsettings += f' - {vbs}\n'
        raise ValueError(f'Invalid boundary setting "{boundary}". \n Valid boundary settings are: \n{valid_bsettings}')

    # check if periodic boundary box dimensions are correct
    if box is not None:
        valid_box(box)
        box = box.astype('float64')

    first_pos, first_triad = _gen_first_pos_and_triad(
                                disc_len,
                                boundary=boundary,
                                box=box,
                                lb=lb,
                                radius=radius,
                                existing_atoms=existing_atoms,
                                first_pos=first_pos,
                                first_triad=first_triad
                                )

    if radius is None or radius == 0:
        # if no excluded volume, no trials are required
        if boundary == 'none':
            return _gen_conf_no_ev(num_monomer,disc_len,lb,first_pos,first_triad)
        if boundary == 'reflecting':
            return _gen_conf_no_ev_reflecting(num_monomer,disc_len,lb,first_pos,first_triad,box)
        if boundary == 'periodic':
            return _gen_conf_no_ev_periodic(num_monomer,disc_len,lb,first_pos,first_triad,box)

    else:
        # with excluded volume the somewhat more expensive generation function is required
        excluded_neighbors = int(np.ceil(radius*2 / disc_len))
        shift_back = int(np.ceil(lb/4/disc_len))

        return _gen_ev_conf(
            num_monomer,
            disc_len,
            lb,
            first_pos,
            first_triad,
            radius,
            excluded_neighbors=excluded_neighbors,
            boundary=boundary,
            box=box,
            existing_atoms=existing_atoms,
            shift_back=shift_back,
            max_trials_per_step=GENPOLYMER_MAXTRIALSPERSTEP)

    raise Exception("Something went wrong in _gen_semiflex")

########################################################################
########################################################################
########################################################################

def _gen_first_pos_and_triad(
    disc_len: float,
    boundary=BOUNDARY_NONE,
    box=None,
    lb=40.0,
    radius=0.0,
    existing_atoms=None,
    first_pos=None,
    first_triad=None,
    generation_depth=0
    ) -> Tuple[np.ndarray]:

    # check if number of repeated attemps exceeds maximum
    if generation_depth>=GENPOLYMER_MAXGENERATION_DEPTH:
        raise ExceedingTrialLimit(GENPOLYMER_MAXGENERATION_DEPTH)
        # raise Exception(f"Number of trial generations for first position and triad exceeded maximum ({GENPOLYMER_MAXGENERATION_DEPTH}).")

    # set number of straight segments generated in direction of tangent
    num_avoid = np.max([1,int(np.ceil(lb/3/disc_len))])
    if radius == 0:
        num_avoid = 0
    
    print('TODO: LIMIT num_avoid to number of monomers to allow for large persistence lengths!')

    # Generate first pos
    if first_pos is None:
        if boundary == BOUNDARY_NONE:
            _first_pos = np.zeros(3)
        else:
            _first_pos = gen_random_point_in_box(box)
    else:
        _first_pos = np.copy(first_pos)

    # Generate first triads (sets orientation of first segment)
    if first_triad is None:
        theta = np.random.random(3) * np.pi
        _first_triad = get_rot_mat(theta)
    else:
        _first_triad = get_rot_mat(theta)

    # Check if first pos violates excluded volumes
    # if existing_atoms is not None and check_ev_violation_single(_first_pos,radius,boundary=boundary,existing_atoms=existing_atoms,box=box):
    if existing_atoms is not None and check_ev_violation_single(_first_pos,radius, existing_atoms, boundary, box=box):
        return _gen_first_pos_and_triad(    disc_len,
                                            boundary=boundary,
                                            box=box,
                                            lb=lb,
                                            radius=radius,
                                            existing_atoms=existing_atoms,
                                            first_pos=first_pos,
                                            first_triad=first_triad,
                                            generation_depth=generation_depth+1
                                            )

    # check if straight segments violate excluded volumes
    for i in range(num_avoid):
        npos = _first_pos + _first_triad[:, 2]*disc_len*(i+1)
        if boundary == 'periodic':
            npos = place_in_box_single(box,npos)
        if boundary == 'reflecting':
            if is_outside_box(npos,box):
                return  _gen_first_pos_and_triad(
                        disc_len,
                        boundary=boundary,
                        box=box,
                        lb=lb,
                        radius=radius,
                        existing_atoms=existing_atoms,
                        first_pos=first_pos,
                        first_triad=first_triad,
                        generation_depth=generation_depth+1
                        )

        # if existing_atoms is not None and check_ev_violation_single(npos,radius,boundary=boundary,existing_atoms=existing_atoms):
        if existing_atoms is not None and check_ev_violation_single(npos,radius, existing_atoms, boundary, box=box):
            return  _gen_first_pos_and_triad(
                    disc_len,
                    boundary=boundary,
                    box=box,
                    lb=lb,
                    radius=radius,
                    existing_atoms=existing_atoms,
                    first_pos=first_pos,
                    first_triad=first_triad,
                    generation_depth=generation_depth+1
                    )
    return _first_pos, _first_triad

########################################################################
########################################################################
########################################################################

# @jit(nopython=True)
@condnumba
def _gen_conf_no_ev(
    num_monomer: int,
    disc_len: float,
    lb: float,
    first_pos: np.ndarray,
    first_triad: np.ndarray
) -> np.ndarray:
    """
    Generate configuration without taking excluded volumes into account
    """
    pos = np.zeros((num_monomer, 3))
    T = np.copy(first_triad)
    sigma = np.sqrt(disc_len / lb)

    pos[0] = first_pos
    pos[1] = first_pos + T[:, 2] * disc_len
    for i in range(2, num_monomer):
        theta = np.random.normal(loc=0.0, scale=sigma, size=3)
        R = get_rot_mat(theta)
        T = np.dot(T, R)
        pos[i] = pos[i - 1] + T[:, 2] * disc_len
    return pos

@condnumba
def _gen_conf_no_ev_reflecting(
    num_monomer: int,
    disc_len: float,
    lb: float,
    first_pos: np.ndarray,
    first_triad: np.ndarray,
    box: np.ndarray,
    generation_depth=0
) -> np.ndarray:
    """
    Generate configuration without taking excluded volumes into account but atoms need to be within volume
    """

    # Generate free configuration
    conf = _gen_conf_no_ev(num_monomer,disc_len,lb,first_pos,first_triad)
    com = np.mean(conf,axis=0)
    bounds = cal_conf_bounds(conf)
    
    for d in range(3):
        conf_drange = bounds[d,1] - bounds[d,0]
        box_drange  = box[d,1] - box[d,0]
        if conf_drange >= box_drange:
            return _gen_conf_no_ev_reflecting(num_monomer,disc_len,lb,first_pos,first_triad,generation_depth=generation_depth+1)
        llim = box[d,0] - bounds[d,0]
        ulim = box[d,1] - bounds[d,1]
        shift = np.random.uniform(llim,ulim)
        for i in range(len(conf)):
            conf[i,d] += shift
    return conf

########################################################################
########################################################################
########################################################################

@condnumba
def _gen_conf_no_ev_periodic(
    num_monomer: int,
    disc_len: float,
    lb: float,
    first_pos: np.ndarray,
    first_triad: np.ndarray,
    box: np.ndarray
    ) -> np.ndarray:
    conf = _gen_conf_no_ev(num_monomer,disc_len,lb,first_pos,first_triad)
    return place_in_box(box,conf)

########################################################################
########################################################################
########################################################################

@condnumba
def cal_conf_bounds(conf: np.ndarray) -> np.ndarray:
    com = np.mean(conf,axis=0)
    bounds = np.zeros((3,2))
    bounds[:,0] = com
    bounds[:,1] = com
    for i in range(len(conf)):
        for d in range(3):
            if conf[i,d] < bounds[d,0]:
                bounds[d,0] = conf[i,d]
            if conf[i,d] > bounds[d,1]:
                bounds[d,1] = conf[i,d]
    return bounds

########################################################################
########################################################################
########################################################################

# @condnumba
def _gen_ev_conf(
    num_monomer: int,
    disc_len: float,
    lb: float,
    first_pos: np.ndarray,
    first_triad: np.ndarray,
    radius: float,
    excluded_neighbors=0,
    boundary='none',
    box=None,
    existing_atoms=None,
    shift_back=20,
    max_trials_per_step=1000
    ) -> np.ndarray:
    """
    Generate configuration considering exlcluded volumes and the periodicity of the box
    """
    ev_size  = radius * 2

    if existing_atoms is not None:
        existing_pos    = existing_atoms[:,:3]
        ev_dists        = existing_atoms[:,3] + radius

    pos = np.zeros((num_monomer, 3))
    triads = np.zeros((num_monomer - 1, 3, 3))

    # T           = np.copy(first_triad)5
    triads[0] = first_triad
    sigma = np.sqrt(disc_len / lb)

    pos[0] = first_pos
    pos[1] = first_pos + first_triad[:, 2] * disc_len
    i = 2

    max_trials = max_trials_per_step * num_monomer
    trials = 0

    while i < num_monomer:
        trials += 1
        if trials >= max_trials:
            raise ExceedingTrialLimit(max_trials)

        theta = np.random.normal(loc=0.0, scale=sigma, size=3)
        R = get_rot_mat(theta)
        triads[i - 1] = np.dot(triads[i - 2], R)
        pos[i] = pos[i - 1] + triads[i - 1, :, 2] * disc_len

        if boundary == 'none':
            if (i - excluded_neighbors) > 0:
                if ev_violation_single(
                    pos[: i - excluded_neighbors], pos[i], ev_size
                ):
                    # print(f"reverting at step {i}")
                    # violated
                    i = i - shift_back
                    if i < 2:
                        i = 2
                    continue
        
        elif boundary == 'reflecting':
            if is_outside_box(pos[i],box):
                # print(f"reverting at step {i}: is outside box")
                i = i - shift_back
                if i < 2:
                    i = 2
                continue

            if (i - excluded_neighbors) > 0:
                if ev_violation_single(
                    pos[: i - excluded_neighbors], pos[i], ev_size
                ):
                    # print(f"reverting at step {i}: violation with previous")
                    # violated
                    i = i - shift_back
                    if i < 2:
                        i = 2
                    continue

        elif boundary == 'periodic':
            pos[i] = place_in_box_single(box, pos[i])
            if (i - excluded_neighbors) > 0:
                if ev_violation_single_periodic_box(
                    pos[: i - excluded_neighbors], pos[i], ev_size, box
                ):
                    # print(f"reverting at step {i}: violation with previous")
                    # violated
                    i = i - shift_back
                    if i < 2:
                        i = 2
                    continue
        
        if existing_atoms is not None:
            # print('check with existing')   
            # print('existing_pos')          
            # print(existing_pos)
            # print('ev_dists')
            # print(ev_dists)
            if boundary == 'periodic':
                if ev_violation_periodic_box_individual_ev_dist(existing_pos ,pos[i],ev_dists, box):
                    # print(f"reverting at step {i}: violation with existing")
                    # violated
                    i = i - shift_back
                    if i < 2:
                        i = 2
                    continue
            else:

                if ev_violation_individual_ev_dist(existing_pos,pos[i],ev_dists):
                    # print(f"reverting at step {i}: violation with existing")
                    # violated
                    i = i - shift_back
                    if i < 2:
                        i = 2
                    continue
        i += 1
    return pos

########################################################################
########################################################################
########################################################################

class ExceedingTrialLimit(Exception):

    def __init__(self, max_trials, *args):
        super().__init__(args)
        self.max_trials = max_trials

    def __str__(self):
        return f'Reached maximum number of trials ({self.max_trials}).'

########################################################################
########################################################################
########################################################################
'''

def plot_polymer_conf(conf: np.ndarray, disc_len=None):
    conflist = list()
    connect = list()
    if disc_len is not None:
        sid = 0
        for i in range(1, len(conf)):
            # print(np.linalg.norm(conf[i]-conf[i-1]))
            if np.abs(np.linalg.norm(conf[i] - conf[i - 1]) - disc_len) > 1e-8:
                pconf = conf[sid:i]
                conflist.append(pconf)
                connect.append(conf[i - 1 : i + 1])
                sid = i
        if sid < len(conf) - 1:
            pconf = conf[sid:]
            conflist.append(pconf)
    else:
        conflist.append(conf)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    for con in connect:
        ax.plot(
            con[:, 0], con[:, 1], con[:, 2], zdir="z", lw=2, alpha=0.3, color="blue"
        )

    for pconf in conflist:
        ax.scatter(
            pconf[:, 0],
            pconf[:, 1],
            pconf[:, 2],
            zdir="z",
            s=50,
            c="black",
            depthshade=True,
        )
        ax.plot(pconf[:, 0], pconf[:, 1], pconf[:, 2], zdir="z", c="black", lw=3)
    plt.show()



def plot_polymer_confs(confs: List[np.ndarray], disc_len=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    colors = plt.cm.viridis(np.linspace(0, 1,len(confs)))

    for cid,conf in enumerate(confs):
        conflist = list()
        connect = list()
        if disc_len is not None:
            sid = 0
            for i in range(1, len(conf)):
                # print(np.linalg.norm(conf[i]-conf[i-1]))
                if np.abs(np.linalg.norm(conf[i] - conf[i - 1]) - disc_len) > 1e-8:
                    pconf = conf[sid:i]
                    conflist.append(pconf)
                    connect.append(conf[i - 1 : i + 1])
                    sid = i
            if sid < len(conf) - 1:
                pconf = conf[sid:]
                conflist.append(pconf)
        else:
            conflist.append(conf)

        for con in connect:
            ax.plot(
                con[:, 0], con[:, 1], con[:, 2], zdir="z", lw=2, alpha=0.3, color="blue"
            )

        for pconf in conflist:
            ax.scatter(
                pconf[:, 0],
                pconf[:, 1],
                pconf[:, 2],
                zdir="z",
                s=200,
                color=colors[cid],
                depthshade=True,
                alpha=0.5
            )
            ax.plot(pconf[:, 0], pconf[:, 1], pconf[:, 2], zdir="z", c="black", lw=3)
    plt.show()
'''

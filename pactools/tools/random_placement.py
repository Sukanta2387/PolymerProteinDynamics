#!/bin/env python3

import sys
from pactools.tools.conditional_numba   import conditional_numba as condnumba
from pactools.tools.excluded_volume     import ev_violation_periodic_box_individual_ev_dist, ev_violation_individual_ev_dist

try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit("terminated")


def random_placement(radius: float, box: np.ndarray, boundary='periodic',existing_atoms=None,trials_per_atom=25,max_penetration=0.0,penetration_steps=11):
    """
        Randomly places an atom of given radius on volume specified by box. The boundary condition may be set to "periodic", "reflecting" or "none".       
        Excluded volume violations can be avoided by providing the list of existing atom positions and their radii (Nx4 np.ndarray).  

        Arguments:

            radius: float
                radius of atom

            box: np.ndarray (3x2)
                Box limits

            boundary: str
                Boundary condition may be set to "periodic", "reflecting" or "none"

            existing_atoms: np.ndarray (default: None)
                Positions and radius of N atoms (Nx4). If set to None, no excluded volume will be considered

            trials_per_atom: int (default 25)
                Number of trial generations per existing atom (total trial: N*trials_per_atom)

            max_penetration: float (default 0.0)
                If the algorithm fails to find a suitable position for the new atom, the radius of the atoms will be reduced to a fraction of the 
                full radius (effectively increasing the penetration depth). This reduction occures gradually in penetration_steps steps to a 
                minimum fraction specified by max_penetration.

            penetration_steps: int (default: 11)
                Number of steps lowering the atom radii (or equivalently inceasing the penetration depth).

        Returns: np.ndarray -> generated position of atom
    """
    genrange = np.copy(box)
    # if boundary == 'reflecting':
    #     genrange[:,0] + 2*radius
    #     genrange[:,1] - 2*radius

    if existing_atoms is None or len(existing_atoms) == 0:
        return gen_random_point_in_box(genrange)

    trials = len(existing_atoms)*trials_per_atom

    if max_penetration < 1.0:
        dist_scales = np.linspace(1.0,max_penetration,penetration_steps)
    else:
        dist_scales = [1.0]


    for dist_scale in dist_scales:
        if dist_scale < 1:
            print(f'dist_scale = {dist_scale}')
        atoms = np.copy(existing_atoms)
        atoms[:,3] = ( atoms[:,3] + radius ) * dist_scale
        r = _random_placement_with_ev(genrange, box, atoms, trials, boundary)
        if r is not None:
            return r
    
    raise Exception(f"Exceeded number of trial placements. Appropriate placement could not be found. try setting the penetration depth to a lower value (currently {max_penetration}).")
    
@condnumba
def gen_random_point_in_box(box: np.ndarray) -> np.ndarray:
    return box[:, 0] + np.random.rand(3) * ( box[:, 1] - box[:, 0] )

def _random_placement_with_ev(genrange: np.ndarray, box: np.ndarray, atoms: np.ndarray, trials: int, boundary: str) -> np.ndarray:
    R        = atoms[:,:3]
    ev_dists = atoms[:,3]
    for trial in range(trials):
        r = gen_random_point_in_box(genrange)
        if boundary == 'periodic':
            if not ev_violation_periodic_box_individual_ev_dist(R,r,ev_dists,box):
                # print(f'period {trial}')
                return r
        else:
            if not ev_violation_individual_ev_dist(R, r, ev_dists):
                # print(f'nonp {trial}')
                return r
    return None










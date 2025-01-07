#!/bin/env python3

import sys
from typing import List,Tuple

from pactools.tools.conditional_numba import conditional_numba as condnumba
# from numba.experimental import jitclass

try:
    import numpy as np
except ModuleNotFoundError:
    print('numpy not installed. Please install numpy')
    sys.exit("terminated")

########################################################################
########################################################################
########################################################################

# specs_exvol = {}
# specs_exvol['bounds']  = nb.types.List(nb.types.List(nb.float))

class ExVol:

    """
        currently only implemented for 3d!
    """

    dims: int
    box: np.ndarray
    boundary: str
    periodic_boundary: bool
    max_diameter: float
    ndia: float

    Ls:  np.ndarray
    blo: np.ndarray
    Ns:  np.ndarray

    cubes: List[List[List[List[np.ndarray]]]]
    num_atoms: int

    def __init__(self,box: np.ndarray, boundary: str, max_diameter: float, ndia = 2.0):

        self.box = box
        self.dims = len(box)
        self.boundary = boundary
        if self.boundary == 'periodic':
            self.periodic_boundary = True
        else:
            self.periodic_boundary = False
        self.max_diameter = max_diameter
        self.ndia = ndia
        self._set_cubes()

        self.num_atoms = 0
        self.all_atoms = list()

        num_cubes = 1
        for d in range(self.dims):
            num_cubes *= self.Ns[d]
        # print(f'ExVol initialized {num_cubes} cubes.')


    def get_neighbor_ids(self,atomid,check_id=True):
        if check_id:
            atom = self._get_atom_with_id(atomid)
        else:
            atom = self.all_atoms[atomid]
        adj_cubes = self.get_adjacent_cubes(atom[:3])
        ids = list()
        for adj_cube in adj_cubes:
            for atom in adj_cube:
                if atom[4] != atomid:
                    ids.append(int(atom[4]))
        return ids

    def check_overlap(self,pos: np.ndarray,radius: float, dist_scale=1) -> bool:
        adj_cubes = self.get_adjacent_cubes(pos)
        for adj_cube in adj_cubes:
            for atom in adj_cube:
                # if check_overmap(pos, adj_cube, radius, dist_scale):
                #     return True
                mdist = (radius + atom[3])*dist_scale
                dist  = np.linalg.norm(pos-atom[:3])
                if dist <= mdist:
                    return True
        return False
    
    def check_internal_overlap(self,dist_scale: float=1) -> bool:
        overlap = False
        for atom1 in self.all_atoms:
            if atom1[4] <= 249:
                continue

            adj_cubes = self.get_adjacent_cubes(atom1[:3])
            for adj_cube in adj_cubes:
                for atom in adj_cube:
                    if atom[4] == atom1[4]:
                        continue
                    # if check_overmap(pos, adj_cube, radius, dist_scale):
                    #     return True
                    mdist = (atom1[3] + atom[3])*dist_scale
                    dist  = np.linalg.norm(atom1[:3]-atom[:3])
                    if dist <= mdist:
                        print('############')
                        print(f'{atom1[4]} {atom[4]}')
                        print(f'{atom1[:3]}')
                        print(f'{atom[:3]}')
                        print (dist)
                        overlap = True
        return overlap

    def get_adjacent_cubes(self,pos: np.ndarray) -> List[List]:
        cubeid = self._pos2cubeid(pos)

        bounds = np.zeros((self.dims,2),dtype=int)
        for d in range(self.dims):
            if cubeid[d] == 0 and not self.periodic_boundary:
                bounds[d,0] = 0
            else:
                bounds[d,0] = cubeid[d] - 1
            
            if cubeid[d] == self.Ns[d]-1 and not self.periodic_boundary:
                bounds[d,1] = cubeid[d]
            else:
                bounds[d,1] = cubeid[d] + 1
        
        if len(bounds) != 3:
            raise Exception("Not implemented for dimensions other than 3.")
        
        cubes = list()
        for cx in range(bounds[0,0],bounds[0,1]+1):
            cxm = cx % self.Ns[0]
            for cy in range(bounds[1,0],bounds[1,1]+1):
                cym = cy % self.Ns[1]
                for cz in range(bounds[2,0],bounds[2,1]+1):
                    czm = cz % self.Ns[2]
                    cubes.append(self.cubes[cxm][cym][czm])
        return cubes

    def add_atoms(self,conf: np.ndarray, radius: float):
        for pos in conf:
            self.add_atom(pos,radius)

    def add_atom(self,pos: np.ndarray, radius: float, atomid=-1) -> List:
        atom = np.zeros(5)
        atom[:3] = pos
        atom[3]  = radius
        if atomid == -1:
            atom[4]  = self.num_atoms
        else:
            atom[4]  = atomid
        cubeid   = self._pos2cubeid(pos)
        cubelist = self._get_cube_elem(cubeid)
        cubelist.append(atom)
        self.num_atoms += 1
        self.all_atoms.append(atom)
        return cubelist
    
    def _get_cube_elem(self, cubeid: np.ndarray) -> List:
        if len(cubeid) != self.dims:
            raise ValueError("Dimension of cubeid does not match the dimension of cubes")
        partial = self.cubes
        for d,cid in enumerate(cubeid):
            if cid == self.Ns[d]:
                cid = 0
            partial = partial[cid]
        return partial
    
    def _pos2cubeid(self,pos) -> np.ndarray:
        ids = np.zeros(self.dims,dtype=int)
        for d in range(self.dims):
            did = int(np.floor((pos[d]-self.blo[d])/self.Ls[d]*self.Ns[d]))
            ids[d] = did
        return ids
    
    def _set_cubes(self) -> None:
        self.Ls  = self.box[:,1] - self.box[:,0]
        self.blo = self.box[:,0]

        self.Ns = np.zeros(self.dims,dtype=np.int32)
        size = self.ndia*self.max_diameter
        for d in range(self.dims):
            self.Ns[d] = int(np.floor(self.Ls[d]/size))
            if self.Ns[d] == 0:
                self.Ns[d] = 1
        self.cubes = self._add_lists(self.Ns)
        
    def _add_lists(self,dims) -> List:
        lists = list()
        for ii in range(dims[0]):
            if len(dims) == 1:
                lists.append(list())
            else:
                lists.append(self._add_lists(dims[1:]))
        return lists
    
    def _get_atom_with_id(self,atomid):
        for atom in self.all_atoms:
            if atom[4] == atomid:
                return atom
        return None
    
@condnumba
def check_overmap(pos: np.ndarray, atoms: np.ndarray, radius: float, dist_scale: float):
    for atom in atoms:
        mdist = (radius + atom[3])*dist_scale
        dist  = np.linalg.norm(pos-atom[:3])
        if dist <= mdist:
            return True
    return False




if __name__ == "__main__":

    box = np.array([[0,10],[0,10],[0,4]])
    boundary = 'periodic'
    max_diameters = 1
    ndia = 2

    exvol = ExVol(box,boundary,max_diameters,ndia=ndia)


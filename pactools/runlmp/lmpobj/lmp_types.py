#!/bin/env python3

import os
import builtins
import sys
from typing import List, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

########################################################################
########################################################################
########################################################################

# class LmpSetup:

#     dimensions  = 3
#     units       = 'lj'
#     boundary    = 'ppp'
#     temp        = 1
#     gamma       = 1
#     atom_style  = 'angle'

#     seed: int

#     def __init__(self,simbox,seed=-1):
#         if seed == -1:
#             self.gen_seed()
#             print(self.seed)

#     ######################################
#     # seed functions
#     def gen_seed(self):
#         self.seed = np.random.randint(0,1000000000)


########################################################################
########################################################################
########################################################################

class AtomType:

    name: str
    id: int
    radius: float
    mass: float

    initialized = False

    def __init__(self, name: str, mass: float, radius: float, id=-1):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.id = id
        if self.id >= 0:
            self.initialized = True
        else:
            self.intialized = False

    def set_id(self, id: int):
        self.id = id
        self.initialized = True

########################################################################
########################################################################
########################################################################

class InteractionType:
    name: str
    id: int
    style: str
    coeffs: List[float]

    initialized = False

    def __init__(self, name: str, style: str, coeffs: List[float], id=-1):
        self.name  = name
        self.style = style
        self.coeffs = coeffs
        self.id = id
        if self.id >= 0:
            self.initialized = True
        else:
            self.intialized = False

    def set_id(self, id: int):
        self.id = id
        self.initialized = True

########################################################################
########################################################################
########################################################################

class BondType(InteractionType):
    def __init__(self, name: str, style: str, coeffs: List[float], id=-1):
        super().__init__(name,style,coeffs,id=id)

########################################################################
########################################################################
########################################################################

class AngleType(InteractionType):
    def __init__(self, name: str, style: str, coeffs: List[float], id=-1):
        super().__init__(name,style,coeffs,id=id)

########################################################################
########################################################################
########################################################################

class DihedralType(InteractionType):
    def __init__(self, name: str, style: str, coeffs: List[float], id=-1):
        super().__init__(name,style,coeffs,id=id)

########################################################################
########################################################################
########################################################################

class ImproperType(InteractionType):
    def __init__(self, name: str, style: str, coeffs: List[float], id=-1):
        super().__init__(name,style,coeffs,id=id)

########################################################################
########################################################################
########################################################################

class Pair:
    atom_name1: str
    atom_name2: str
    style: str
    cutoff: float
    coeffs: List[float]
    atomtype_id1: int
    atomtype_id2: int

    atomtype1: AtomType
    atomtype2: AtomType

    initialized = False

    def __init__(
        self,
        atom_name1: str,
        atom_name2: str,
        style: str,
        cutoff: float,
        coeffs: List[float],
        atomtype_id1=-1,
        atomtype_id2=-1,
        atomtype1=None,
        atomtype2=None,
    ):
        self.atom_name1 = atom_name1
        self.atom_name2 = atom_name2
        self.style = style
        self.cutoff = cutoff
        self.coeffs = [coeff for coeff in coeffs]
        self.atomtype_id1 = atomtype_id1
        self.atomtype_id2 = atomtype_id2
        self.atomtype1 = atomtype1
        self.atomtype2 = atomtype2

        if self.atomtype1 is not None:
            self._check_conproblemsistent_atomtype(self.atomtype1)
            if self.atomtype1.initialized:
                self.atomtype_id1 = self.atomtype1.id
        if self.atomtype2 is not None:
            self._check_consistent_atomtype(self.atomtype2)
            if self.atomtype2.initialized:
                self.atomtype_id2 = self.atomtype2.id
        self.check_initialized()

    def _check_consistent_atomtype(self, atomtype):
        if type(atomtype) is not AtomType:
            raise TypeError(
                "Error in constructor of Pair: atomtype is not of type AtomType!"
            )

    def check_initialized(self):
        if self.initialized:
            return True
        if self.atomtype_id1 < 0:
            return False
        if self.atomtype_id2 < 0:
            return False
        if self.atomtype1 is None:
            return False
        if self.atomtype2 is None:
            return False
        self.initialized = True
        return True

    def set_atomids(self, id1: int, id2: int):
        self.atomtype_id1 = id1
        self.atomtype_id2 = id2
        self.check_initialized()

    def set_atomid(self, atomname: str, atomid: int):
        if self.atom_name1 == atomname:
            self.atomtype_id1 = atomid
        if self.atom_name2 == atomname:
            self.atomtype_id2 = atomid
        self.check_initialized()

    def set_atomtypes(self, atomtype1: AtomType, atomtype2: AtomType):
        self._check_consistent_atomtype(atomtype1)
        self._check_consistent_atomtype(atomtype2)
        self.atomtype1 = atomtype1
        self.atomtype2 = atomtype2
        if self.atomtype1.initialized:
            self.atomtype_id1 = self.atomtype1.id
        if self.atomtype2.initialized:
            self.atomtype_id2 = self.atomtype2.id
        self.check_initialized()

    def set_atomtype(self, atomtype: AtomType):
        if self.atom_name1 == atomtype.name:
            self.atomtype1 = atomtype
            self._check_consistent_atomtype(atomtype)
            if self.atomtype1.initialized:
                self.atomtype_id1 = atomtype.id
        if self.atom_name2 == atomtype.name:
            self.atomtype2 = atomtype
            self._check_consistent_atomtype(atomtype)
            if self.atomtype2.initialized:
                self.atomtype_id2 = atomtype.id
        self.check_initialized()

    def __eq__(self, otherpair):
        if not isinstance(otherpair, Pair):
            return False
        if self.style != otherpair.style:
            return False
        if self.cutoff != otherpair.cutoff:
            return False
        if len(self.coeffs) != len(otherpair.coeffs):
            return False
        for i in range(len(self.coeffs)):
            if self.coeffs[i] != otherpair.coeffs[i]:
                return False
        return True

########################################################################
########################################################################
########################################################################

class MoleculeType:
    name:   str
    id:     int

    bondtypes = dict()
    bondtypes_list: List[BondType]

    angletypes = dict()
    angletypes_list: List[AngleType]
    
    initialized = False

    def __init__(self, name: str, bondtypes=None, angletypes=None, id=-1):
        self.name = name
        self.set_id(id)
        self.set_bondtypes(bondtypes)
        self.set_angletypes(angletypes)

    ########################################################################

    def set_id(self, id: int):
        self.id = id
        if self.id > 0:
            self.initialized = True

    ########################################################################

    def set_bondtypes(self,bondtypes=None):
        self.set_interactiontypes(self,self.bondtypes,self.bondtypes_list,interactiontypes=bondtypes,interactiontypes_name='bondtypes')

    def add_bondtype(self,bondtype: BondType):
        self.add_interactiontype(self,self.bondtypes,self.bondtypes_list,bondtype,interactiontypes_name='bondtype')

    def set_angletypes(self,angletypes=None):
        self.set_interactiontypes(self,self.angletypes,self.angletypes_list,interactiontypes=angletypes,interactiontypes_name='angletypes')

    def add_angletype(self,angletype: AngleType):
        self.add_interactiontype(self,self.angletypes,self.angletypes_list,angletype,interactiontypes_name='angletype')

    ########################################################################
    ########################################################################

    def set_interactiontypes(self,self_interactiontypes: dict,self_interactiontypes_list: list, interactiontypes=None, interactiontypes_name='interactiontypes'):
        if interactiontypes is not None:
            if isinstance(interactiontypes) == dict:
                self_interactiontypes       = interactiontypes
                self_interactiontypes_list  = list(set([interactiontypes[key] for key in interactiontypes.keys()]))
            elif isinstance(interactiontypes) == list:
                self_interactiontypes_list = list(set(interactiontypes))
                self_interactiontypes      = dict()
                for interactiontype in self_interactiontypes_list:
                    self_interactiontypes[interactiontype.name] = interactiontype
            else:
                raise TypeError(f"{interactiontypes_name} needs to be either a dictionary or a list.")

    def add_interactiontype(self,self_interactiontypes,self_interactiontyles_list,interactiontype,interactiontypes_name='interactiontypes'):
        if interactiontype.name in self_interactiontypes.keys():
            print(f"{interactiontypes_name} '{interactiontype.name}' was already contained in molecule {self.name}.")
        self_interactiontypes[interactiontype.name] = interactiontype
        self_interactiontyles_list.append(interactiontype)

    ########################################################################


########################################################################
########################################################################
########################################################################

if __name__ == "__main__":

    pass

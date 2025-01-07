#!/bin/env python3

import os
import sys
from typing import List
from .lmp_types import AtomType, BondType, AngleType, DihedralType, ImproperType

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")


########################################################################
########################################################################
########################################################################

class NoMolecule:
    def __init__(self):
        self.name = 'nomol'
        self.id   = 0

nomol = NoMolecule()


class Atom:
    
    type:       AtomType
    position:   np.ndarray
    velocity:   np.ndarray
    id:         int

    # molecule: Molecule

    def __init__(self, type: AtomType, position: np.ndarray, velocity=None, id=-1, molecule=None):
        self.type = type
        self.position = position
        self.velocity = None
        self.id = id
        if molecule is None:
            self.molecule = nomol
        else:
            self.molecule = molecule

    def set_id(self, id: int):
        self.id = id
    
    def set_molecule(self,molecule=None):
        self.molecule = molecule
    
    def set_position(self,position: np.ndarray):
        self.position = position

    def get_molecule_id(self) -> int:
        if self.molecule is None:
            return 0
        return self.molecule.id
    
########################################################################
########################################################################
########################################################################

class Bond:
    type:       BondType
    atomids:    List[int]
    id:         int

    def __init__(self, type: BondType, atomids: List[int], id=-1):
        self.type = type
        self.atomids = atomids
        self.id = id

    def set_id(self, id: int):
        self.id = id
    
    def set_atomids(self,atomids: List[int]):
        self.atomids = atomids

########################################################################
########################################################################
########################################################################

class Angle:
    type:       AngleType
    atomids:    List[int]
    id:         int

    def __init__(self, type: AngleType, atomids: List[int], id=-1):
        self.type = type
        self.atomids = atomids
        self.id = id

    def set_id(self, id: int):
        self.id = id
    
    def set_atomids(self,atomids: List[int]):
        self.atomids = atomids

########################################################################
########################################################################
########################################################################

class Dihedral:
    type:       DihedralType
    atomids:    List[int]
    id:         int

    def __init__(self, type: DihedralType, atomids: List[int], id=-1):
        self.type = type
        self.atomids = atomids
        self.id = id

    def set_id(self, id: int):
        self.id = id
    
    def set_atomids(self,atomids: List[int]):
        self.atomids = atomids

########################################################################
########################################################################
########################################################################

class Improper:
    type:       ImproperType
    atomids:    List[int]
    id:         int

    def __init__(self, type: ImproperType, atomids: List[int], id=-1):
        self.type    = type
        self.atomids = atomids
        self.id      = id

    def set_id(self, id: int):
        self.id = id
    
    def set_atomids(self,atomids: List[int]):
        self.atomids = atomids

########################################################################
########################################################################
########################################################################

class Molecule:
    name:       str

    atoms:      List[Atom]
    bonds:      List[Bond]
    angles:     List[Angle]
    dihedrals:  List[Dihedral]
    impropers:  List[Improper]

    id:         int

    def __init__(self, name: str, atoms: List[Atom], bonds=[], angles=[], dihedrals=[], impropers=[], id=-1 ):
        self.name  = name

        self._init_lists()

        self.add_atoms(atoms)
        self.add_bonds(bonds)
        self.add_angles(angles)
        self.add_dihedrals(dihedrals)
        self.add_impropers(impropers)

        self.id = id

    def set_id(self, id: int) -> None:
        self.id = id

    def _init_lists(self) -> None:
        self.atoms = list()
        self.bonds = list()
        self.angles = list()
        self.dihedrals = list()
        self.impropers = list()


    ########################################################################

    def add_atom(self,atom: Atom, sort=True) -> None:
        if atom not in self.atoms:
            self.atoms.append(atom)
            atom.set_molecule(self)
            if sort:
                self._sort_atoms()
    
    def add_atoms(self,atoms: List[Atom], sort=True) -> None:
        for atom in atoms:
            self.add_atom(atom,sort=False)
        if sort:
            self._sort_atoms()

    def remove_atom(self,atom=None,atomid=None) -> None:
        if atom is None and atomid is not None:
            for i in range(len(self.atoms)):
                if self.atoms[i].id == atomid:
                    atom = self.atoms[i]
                    break
        if atom is not None:
            self.atoms.remove(atom)
            
    def remove_atoms(self,atoms=[],atomids=[]) -> None:
        if len(atoms) > 0:
            for atom in atoms:
                self.remove_atom(atom=atom)
        else:
            for aid in atomids:
                self.remove_atom(atomid=aid)

    def get_atom_ids(self) -> List[int]:
        return sorted([atom.id for atom in self.atoms])

    def get_atoms(self) -> List[Atom]:
        return self.atoms

    def get_atom_by_id(self,id: int) -> Atom:
        for atom in self.atoms:
            if atom.id == id:
                return atom
        return None
    
    ########################################################################

    def add_bond(self,bond: Bond, sort=True):
        if bond not in self.bonds:
            self.bonds.append(bond)
            if sort:
                self._sort_bonds()
    
    def add_bonds(self,bonds: List[Bond], sort=True):
        for bond in bonds:
            self.add_bond(bond,sort=False)
        if sort:
            self._sort_bonds()
    
    def remove_bond(self,bond=None,bondid=None):
        if bond is None and bondid is not None:
            for i in range(len(self.bonds)):
                if self.bonds[i].id == bondid:
                    bond = self.bonds[i]
                    break
        if bond is not None:
            self.bonds.remove(bond)
            
    def remove_bonds(self,bonds=[],bondids=[]):
        if len(bonds) > 0:
            for bond in bonds:
                self.remove_bond(bond=bond)
        else:
            for aid in bondids:
                self.remove_bond(bondid=aid)

    def get_bond_ids(self) -> List[int]:
        return sorted([bond.id for bond in self.bonds])

    def get_bonds(self) -> List[Bond]:
        return self.bonds

    def get_bond_by_id(self,id: int) -> Bond:
        for bond in self.bonds:
            if bond.id == id:
                return bond
        return None

    ########################################################################

    def add_angle(self,angle: Angle, sort=True):
        if angle not in self.angles:
            self.angles.append(angle)
            if sort:
                self._sort_angles()
    
    def add_angles(self,angles: List[Angle], sort=True):
        for angle in angles:
            self.add_angle(angle,sort=False)
        if sort:
            self._sort_angles()

    def remove_angle(self,angle=None,angleid=None):
        if angle is None and angleid is not None:
            for i in range(len(self.angles)):
                if self.angles[i].id == angleid:
                    angle = self.angles[i]
                    break
        if angle is not None:
            self.angles.remove(angle)
            
    def remove_angles(self,angles=[],angleids=[]):
        if len(angles) > 0:
            for angle in angles:
                self.remove_angle(angle=angle)
        else:
            for aid in angleids:
                self.remove_angle(angleid=aid)

    def get_angle_ids(self) -> List[int]:
        return sorted([angle.id for angle in self.angles])

    def get_angles(self) -> List[Angle]:
        return self.angles

    def get_angle_by_id(self,id: int) -> Angle:
        for angle in self.angles:
            if angle.id == id:
                return angle
        return None
    
    ########################################################################

    def add_dihedral(self,dihedral: Dihedral, sort=True):
        if dihedral not in self.dihedrals:
            self.dihedrals.append(dihedral)
            if sort:
                self._sort_dihedrals()
    
    def add_dihedrals(self,dihedrals: List[Dihedral], sort=True):
        for dihedral in dihedrals:
            self.add_dihedral(dihedral,sort=False)
        if sort:
            self._sort_dihedrals()

    def remove_dihedral(self,dihedral=None,dihedralid=None):
        if dihedral is None and dihedralid is not None:
            for i in range(len(self.dihedrals)):
                if self.dihedrals[i].id == dihedralid:
                    dihedral = self.dihedrals[i]
                    break
        if dihedral is not None:
            self.dihedrals.remove(dihedral)
            
    def remove_dihedrals(self,dihedrals=[],dihedralids=[]):
        if len(dihedrals) > 0:
            for dihedral in dihedrals:
                self.remove_dihedral(dihedral=dihedral)
        else:
            for aid in dihedralids:
                self.remove_dihedral(dihedralid=aid)

    def get_dihedral_ids(self) -> List[int]:
        return sorted([dihedral.id for dihedral in self.dihedrals])

    def get_dihedrals(self) -> List[Dihedral]:
        return self.dihedrals

    def get_dihedral_by_id(self,id: int) -> Dihedral:
        for dihedral in self.dihedrals:
            if dihedral.id == id:
                return dihedral
        return None
    
    ########################################################################

    def add_improper(self,improper: Improper, sort=True):
        if improper not in self.impropers:
            self.impropers.append(improper)
            if sort:
                self._sort_impropers()
    
    def add_impropers(self,impropers: List[Improper], sort=True):
        for improper in impropers:
            self.add_improper(improper,sort=False)
        if sort:
            self._sort_impropers()
    
    def remove_improper(self,improper=None,improperid=None):
        if improper is None and improperid is not None:
            for i in range(len(self.impropers)):
                if self.impropers[i].id == improperid:
                    improper = self.impropers[i]
                    break
        if improper is not None:
            self.impropers.remove(improper)
            
    def remove_impropers(self,impropers=[],improperids=[]):
        if len(impropers) > 0:
            for improper in impropers:
                self.remove_improper(improper=improper)
        else:
            for aid in improperids:
                self.remove_improper(improperid=aid)

    def get_improper_ids(self) -> List[int]:
        return sorted([improper.id for improper in self.impropers])

    def get_impropers(self) -> List[Improper]:
        return self.impropers

    def get_improper_by_id(self,id: int) -> Improper:
        for improper in self.impropers:
            if improper.id == id:
                return improper
        return None

    ########################################################################

    def _sort_atoms(self):
        self.atoms = sorted(self.atoms, key=lambda a: a.id)

    def _remove_atom_duplicates(self,sort=True):
        self.atoms = list(*set(self.atoms))
        if sort:
            self._sort_atoms()

    def _sort_bonds(self):
        self.bonds = sorted(self.bonds, key=lambda a: a.id)

    def _remove_bond_duplicates(self,sort=True):
        self.bonds = list(*set(self.bonds))
        if sort:
            self._sort_bonds()

    def _sort_angles(self):
        self.angles = sorted(self.angles, key=lambda a: a.id)

    def _remove_angle_duplicates(self,sort=True):
        self.angles = list(*set(self.atanglesbondsoms))
        if sort:
            self._sort_angles()

    def _sort_dihedrals(self):
        self.dihedrals = sorted(self.dihedrals, key=lambda a: a.id)

    def _remove_dihedral_duplicates(self,sort=True):
        self.dihedrals = list(*set(self.dihedrals))
        if sort:
            self._sort_dihedrals()

    def _sort_impropers(self):
        self.impropers = sorted(self.impropers, key=lambda a: a.id)

    def _remove_improper_duplicates(self,sort=True):
        self.impropers = list(*set(self.impropers))
        if sort:
            self._sort_impropers()
    
    def _sort_all(self):
        self._sort_atoms()
        self._sort_bonds()
        self._sort_angles()
        self._sort_dihedrals()
        self._sort_impropers()

    def _remove_duplicates(self):
        self._remove_atom_duplicates()
        self._remove_bond_duplicates()
        self._remove_angle_duplicates()
        self._remove_dihedral_duplicates()
        self._remove_improper_duplicates()

    ########################################################################




########################################################################
########################################################################
########################################################################

if __name__ == "__main__":

    pass

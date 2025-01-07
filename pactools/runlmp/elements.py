#!/bin/env python3

import os
import sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

from pactools.runlmp.lmpobj import Atom, Bond, Angle, Dihedral, Improper, Molecule
from pactools.runlmp        import TypeSetup, TypeNotDefined
from pactools.inout         import write_xyz,read_xyz
from pactools.inout         import read_custom
from pactools.inout         import write_conf
from pactools.tools         import random_placement, gen_random_point_in_box
from pactools.tools         import ExVol


# from .lmpobj.lmp_elements import Atom, Bond, Angle, Dihedral, Improper, Molecule
# from .types import TypeSetup, TypeNotDefined
# from ..inout.xyz import write_xyz,read_xyz
# from ..inout.lmpconf import write_conf
# from ..tools.random_placement import *
# from ..tools.ExVol import ExVol

########################################################################
########################################################################
########################################################################


class Elements:

    types: TypeSetup
    atoms: List[Atom]
    bonds: List[Bond]
    angles: List[Angle]
    dihedrals: List[Dihedral]
    impropers: List[Improper]

    molecules: List[Molecule]
    molecules_dict = dict()

    simulation_box: np.ndarray
    boundary: str
    num_atoms: int

    trials_per_atom = 25
    max_penetration = 0.0
    penetration_steps = 11

    exvol: ExVol
    use_exvol: bool

    ########################################################################
    ############### CONSTRUCTOR ############################################
    ########################################################################

    def __init__(self, simulation_box: np.ndarray, boundary: str, types = None, max_radius=0):
        self.simulation_box = simulation_box
        self.boundary = boundary

        if types is None:
            self.types = TypeSetup()
        else:
            self.types = types

        self.num_atoms = 0
        self.atoms = list()
        self.molecules = list()
        self.bonds = list()
        self.angles = list()
        self.dihedrals = list()
        self.impropers = list()

        self.max_radius = max_radius
        if max_radius > 0:
            self.use_exvol = True
            self.init_exvol()
            # self.exvol = ExVol(self.simulation_box,self.boundary,self.max_radius*2)
        else:

            self.use_exvol = False
        

    def set_placement_trials(
        self, trials_per_atom: int, max_penetration: float, penetration_steps: int
    ):
        self.trials_per_atom = trials_per_atom
        self.max_penetration = max_penetration
        self.penetration_steps = penetration_steps

    def get_positions(self, add_typeid=False, add_radius=False):
        dim = 3
        if add_typeid:
            dim += 1
        if add_radius:
            dim += 1
        pos = np.zeros((len(self.atoms), dim))
        for i, atom in enumerate(self.atoms):
            pos[i, :3] = atom.position
            d = 2
            if add_typeid:
                d += 1
                pos[i, d] = atom.type.id
            if add_radius:
                d += 1
                pos[i, d] = atom.type.radius
        return pos

    ########################################################################
    ############### ADD ATOMS ##############################################
    ########################################################################

    def add_atom(self, typename: str, position=None, ev=True) -> Atom:
        if typename not in self.types.atomtypes.keys():
            raise TypeNotDefined("AtomType",typename)

        type = self.types.atomtypes[typename]
        if position is None:
            if ev:
                if self.use_exvol:
                    position = self.gen_random_position_exvol(type.radius)
                    self.exvol.add_atom(position,type.radius,atomid=len(self.atoms) + 1)
                else:
                    existing_atoms = self.get_positions(add_typeid=False, add_radius=True)
                    position = random_placement(
                        type.radius,
                        self.get_generation_box(),
                        boundary=self.boundary,
                        existing_atoms=existing_atoms,
                        trials_per_atom=self.trials_per_atom,
                        max_penetration=self.max_penetration,
                        penetration_steps=self.penetration_steps,
                    )
            else:
                position = gen_random_point_in_box(self.get_generation_box())
        elif ev and self.use_exvol:
            self.exvol.add_atom(position,type.radius,atomid=len(self.atoms) + 1)
        new_atom = Atom(type, position, id=len(self.atoms) + 1)
        self.atoms.append(new_atom)
        return new_atom

    def add_atoms(self, typename: str, number: int, positions=None, ev=True) -> List[Atom]:
        added_atoms = list()
        if positions is None:
            for i in range(number):
                new_atom = self.add_atom(typename)
                added_atoms.append(new_atom)
        else:
            if len(positions) != number:
                raise ValueError(
                    f"Specified number of atoms to add inconsistent with dimension of positions vector."
                )
            for position in positions:
                new_atom = self.add_atom(typename, position, ev=ev)
                added_atoms.append(new_atom)
        return added_atoms
    
    def gen_random_position_exvol(self, radius: float):

        genrange = self.get_generation_box()
        if self.max_penetration < 1.0:
            dist_scales = np.linspace(1.0,self.max_penetration,self.penetration_steps)
        else:
            dist_scales = [1.0]

        for dist_scale in dist_scales:
            for trial in range(self.trials_per_atom):
                pos = gen_random_point_in_box(genrange)
                if not self.exvol.check_overlap(pos,radius,dist_scale=dist_scale):
                    return pos
        raise Exception(f"Exceeded number of trial placements. Appropriate placement could not be found. try setting the penetration depth to a lower value (currently {self.max_penetration}).")
    
    def init_exvol(self):
        if self.use_exvol:
            self.exvol = ExVol(self.simulation_box,self.boundary,self.max_radius*2)
            for atom in self.atoms:
                self.exvol.add_atom(atom.position,atom.type.radius,atom.id)


        
    ########################################################################
    ############### ADD MOLECULE P##########################################
    ########################################################################

    def add_molecule(self, name: str, atoms: List[Atom], bonds=[], angles=[], dihedrals=[], impropers=[]) -> Molecule:
        if name in self.molecules_dict.keys():
            print(f"Warning: molecule name '{name}' already in use. Adding numeral.")
            basename = str(name)
            numeral = 1
            while name in self.molecules_dict.keys():
                numeral += 1
                name = basename + f'_#{numeral}'

        new_mol = Molecule(name,atoms,bonds=bonds,angles=angles,dihedrals=dihedrals,impropers=impropers,id=len(self.molecules) + 1)
        self.molecules.append(new_mol)
        self.molecules_dict[name] = new_mol
        return new_mol

    ########################################################################
    ############### ADD BOND P##############################################
    ########################################################################

    def add_bond(self, typename: str, atomids: List[int]) -> Bond:
        if typename not in self.types.bondtypes.keys():
            raise TypeNotDefined("BondType",typename)

        type = self.types.bondtypes[typename]
        new_bond = Bond(type, atomids, id=len(self.bonds) + 1)
        self.bonds.append(new_bond)
        return new_bond

    ########################################################################
    ############### ADD ANGLE P#############################################
    ########################################################################

    def add_angle(self, typename: str, atomids: List[int]) -> Angle:
        if typename not in self.types.angletypes.keys():
            raise TypeNotDefined("AngleType",typename)

        type = self.types.angletypes[typename]
        new_angle = Angle(type, atomids, id=len(self.angles) + 1)
        self.angles.append(new_angle)
        return new_angle

    ########################################################################
    ############### ADD DIHEDRAL P##########################################
    ########################################################################

    def add_dihedral(self, typename: str, atomids: List[int]) -> Dihedral:
        if typename not in self.types.dihedraltypes.keys():
            raise TypeNotDefined("DihedralType",typename)

        type = self.types.dihedraltypes[typename]
        new_dihedral = Dihedral(type, atomids, id=len(self.dihedrals) + 1)
        self.dihedrals.append(new_dihedral)
        return new_dihedral

    ########################################################################
    ############### ADD IMPROPER P##########################################
    ########################################################################

    def add_improper(self, typename: str, atomids: List[int]) -> Improper:
        if typename not in self.types.impropertypes.keys():
            raise TypeNotDefined("ImproperType",typename)

        type = self.types.impropertypes[typename]
        new_improper = Improper(type, atomids, id=len(self.impropers) + 1)
        self.impropers.append(new_improper)
        return new_improper
    

    ########################################################################
    ############### BOX GETTER #############################################
    ########################################################################

    def get_generation_box(self):
        if self.boundary == 'reflecting':
            ld = self.types.largest_radius()*2
            box = np.array(self.simulation_box,dtype=np.float64)
            if np.min(box[:,1]-box[:,0]) <= ld*2:
                raise Exception('Simulation box is smaller than atom diameter!')
            box[:,0] += ld 
            box[:,1] -= ld 
            return box
        else:
            return self.simulation_box


    ########################################################################
    ############### PRINT TO XYZ ###########################################
    ########################################################################

    def dump2xyz(self, filename: str,append=False,typedict=None,variable_atoms=None,additional_pos='first',timestep=None) -> None:
        """
            typedict:
                typedict should be a dictionary that translates atomtype names 
                from pactools to atomtype in xyz. 
                This may be useful for quick display of known atom types (like C and O).

            variable_atoms:

        """
        if typedict is None:
            typedict = dict()
            for atomtype in self.types.atomtypes_list:
                typedict[atomtype.name] = atomtype.name
        
        if additional_pos == 'first':
            additional_pos = self.atoms[0].position
        else:
            additional_pos = self.simulation_box[:,-1]
        

        if variable_atoms is None:
            pos = self.get_positions()
            types = [typedict[atom.type.name] for atom in self.atoms]

        else:
            pos   = list()
            types = list()
            for atom in self.atoms:
                if atom.id in variable_atoms.keys():
                    for vtype in variable_atoms[atom.id]:
                        if vtype == atom.type.name:
                            pos.append(atom.position)
                        else:
                            pos.append(additional_pos)
                        types.append(typedict[vtype])
                else:
                    pos.append(atom.position)
                    types.append(typedict[atom.type.name])


        pos = [np.array(pos)]

        # print(pos)
        # print(types)

        # if typedict is None:
        #     types = [atom.type.name for atom in self.atoms]
        # else:
        #     types = [typedict[atom.type.name] for atom in self.atoms]
        data = dict()
        data['pos']   = pos
        data['types'] = types
        if timestep is not None:
            data['timesteps'] = [timestep]
        write_xyz(filename, data, add_extension=True,append=append)

    ########################################################################
    ############### GENERATE CONF FILE #####################################
    ########################################################################

    # def simdict(self) -> dict:
    #     sim = dict()
    #     sim['title']        = 'simdict'
    #     sim['atomtypes']    = self.types.atomtypes_list
    #     sim['bondtypes']    = self.types.bondtypes
    #     sim['angletypes']   = self.types.angletypes_list
    #     sim['box']          = self.simulation_box
    #     sim['atoms']        = self.atoms
    #     sim['bonds']        = self.bonds
    #     sim['angles']       = self.angles
    #     return sim

    def genconf(self, filename: str) -> None:
        write_conf(filename,self)

    ########################################################################
    ############### GENERATE INPUT LINES ###################################
    ########################################################################


    
        
    

    ########################################################################
    ############### READ POSITIONS FROM FILE ###############################
    ########################################################################

    def xyz2pos(self,filename: str,snapshot=-1) -> None:
        data  = read_xyz(filename)
        conf  = data['pos'][snapshot]
        types = data['types']
        if len(types) != len(self.atoms):
            raise ValueError(f"Number of atoms in xyz ({len(types)}) does not match specified number of atoms ({len(self.atoms)})!")

        for i in range(len(self.atoms)):
            # if types[i] != self.atoms[i].type.name:
            #     raise ValueError(f"Atom type mismatch: expecting type '{self.atoms[i].type.name}, but '{types[i]}' was given.")
            self.atoms[i].position = conf[i]

        self.init_exvol()

    def custom2pos(self,filename: str,snapshot=-1) -> None:
        specs = read_custom(filename,sortbyid=True,splitargs=True)
        pos   = specs['position'][snapshot]
        ids   = specs['id'][snapshot]
        if len(ids) != len(self.atoms):
            raise ValueError(f"Number of atoms in custom file ({len(ids)}) does not match specified number of atoms ({len(self.atoms)})!")

        for i,atom in enumerate(self.atoms):
            if atom.id != ids[i]:
                raise ValueError(f"Inconsistent atom ids in custom file")
            atom.position = pos[i]
        if 'velocity' in specs:
            vel = specs['velocity'][snapshot]
            for i,atom in enumerate(self.atoms):
                atom.velocity = vel[i]

        self.init_exvol()


    ########################################################################
    ############### ATOM GETTER METHODS ####################################
    ########################################################################

    def get_atoms(self,ids=None):
        if ids is None:
            return self.atoms
        assert (isinstance(ids,list) or isinstance(ids,np.ndarray)), "ids needs to be of type list or np.ndarray"
        return [self.atoms[id-1] for id in ids]

    def get_atompos(self,ids=None,add_id=False):
        atoms = self.get_atoms(ids=ids)
        if add_id:
            pos = np.zeros((len(atoms),4))
            for i,atom in enumerate(atoms):
                pos[i,:3] = atom.position
                pos[i,3]  = atom.id
        else:
            pos = np.zeros((len(atoms),3))
            for i,atom in enumerate(atoms):
                pos[i] = atom.position
        return pos
    
    def get_atompos_inrange(self, lw: int,up: int):
        return self.get_atompos(ids=np.arange(lw,up))
    
    def get_atomtypes(self,ids=None):
        atoms = self.get_atoms(ids=ids)
        return [atom.type.name for atom in atoms]

    def get_atoms_of_type(self,typename: str):
        return [atom for atom in self.atoms if atom.type.name == typename]

    def get_atoms_of_types(self,typenames: List[str]):
        return [atom for atom in self.atoms if atom.type.name in typenames]

    def get_atompos_of_type(self,typename: str):
        return self.get_atompos(ids=self.get_atomids_of_type(typename))
    
    def get_atompos_of_types(self,typenames: List[str]):
        return self.get_atompos(ids=self.get_atomids_of_types(typenames))
        
    def get_atomids_of_type(self,typename: str):
        return [atom.id for atom in self.atoms if atom.type.name == typename]
    
    def get_atomids_of_types(self,typenames: List[str]):
        return [atom.id for atom in self.atoms if atom.type.name in typenames]
    
    def velocity_set(self):
        for atom in self.atoms:
            if atom.velocity is None:
                return False
        return True

    ########################################################################
    ############### ATOM SETTER METHODS ####################################
    ########################################################################

    def change_atomtype(self,atomid: int, atomtypename: str):
        if atomtypename not in self.types.atomtypes.keys():
            raise ValueError(f"Unknown atomtype '{atomtypename}'")
        self.atoms[atomid-1].type = self.types.atomtypes[atomtypename]

    def set_atompos(self,positions: np.ndarray,ids=None) -> None:
        if ids is not None and len(ids) != len(positions):
            raise ValueError(f'The length of the provided list of ids is inconsistent with the size of the positions.')
        atoms = self.get_atoms(ids=ids)
        for i,atom in enumerate(atoms):
            atom.position = positions[i]
    

########################################################################
########################################################################
########################################################################

if __name__ == "__main__":

    typehandler = TypeSetup()
    typehandler.add_atomtype("protein", 0.5, 1)
    typehandler.add_atomtype("dna1", 0.5, 1)
    typehandler.add_atomtype("dna2", 0.5, 1)

    box = np.array([[0, 50], [0, 50], [0, 50]])

    elems = Elements(typehandler, box, "periodic")

    for i in range(50):
        elems.add_atom("protein")

    for atom in elems.atoms:
        print(atom.id)

    filename = "test/xyz_test"
    elems.print2xyz(filename)

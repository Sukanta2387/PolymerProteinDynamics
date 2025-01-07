#!/bin/env python3

import os
import sys
from typing import List, Union

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

from pactools.tools             import gen_polymer
from pactools.tools             import unwrap_polymer, place_in_box
from pactools.runlmp.elements   import Elements
from pactools.runlmp.types      import TypeSetup, TypeNotDefined
# from .lmpobj.lmp_types import BondType,AngleType,AtomType

########################################################################
########################################################################
########################################################################


ADD_POLYMER_DEFAULT_FENE_K       = 30
ADD_POLYMER_DEFAULT_FENE_EPSILON = 1.0
ADD_POLYMER_DEFAULT_FENE_R0frac  = 1.5
ADD_POLYMER_DEFAULT_HARMONIC_K   = 50

ADD_POLYMER_DEFAULT_ANGLESTYLE   = 'cosine'

ADD_POLYMER_BASENAME      = 'poly'
ADD_POLYMER_BASEBONDNAME  = 'polybond'
ADD_POLYMER_BASEANGLENAME = 'polyangle'

def add_polymer(    elems: Elements, 
                    num_monomer: int, 
                    disc_len: float, 
                    sequence: Union[str,List[str]], 
                    bondtypename=None, 
                    bondstyle='fene', 
                    bend_modulus = 0.0,
                    init_bend_mod = None, 
                    mass = 1,
                    angletype=None,
                    ev=True,
                    name='none',
                    startatcenter=False,
                    centercom=False
                    ):

    """
        If a atomtypename specified in sequence is not yet defined it will be defined in this function. The radius will be half the 
        discretization length (disc_len).
    """

    #############################################
    # Atoms
    #############################################

    # set sequence if only single type provided
    if isinstance(sequence,str):
        sequence = [sequence for i in range(num_monomer)]

    # check if sequence length matches number of monomers
    if len(sequence) != num_monomer:
        raise ValueError("Length of provided sequence does not match num_monomers.")

    # initiate atomtypes and typenames
    atomtype_names = list(set(sequence))
    for name in atomtype_names:
        if not elems.types.atomtype_contained(name):
            elems.types.add_atomtype(name, 0.5*disc_len, mass)
    
    # set persistence length for configuration generation
    if init_bend_mod is None:
        if bend_modulus <= 0:
            lb = disc_len*3
        else:
            lb = bend_modulus
    else:
        lb = init_bend_mod

    # generate the configuration
    existing_atoms = elems.get_positions(add_typeid=False, add_radius=True)
    if len(existing_atoms) == 0:
        existing_atoms = None
    if ev:
        if startatcenter:
            first_pos = (elems.simulation_box[:,0] + elems.simulation_box[:,1])* 0.5
        else:
            first_pos = None  
        conf = gen_polymer(num_monomer,disc_len,elems.boundary,elems.get_generation_box(),lb=lb,radius=disc_len*0.5,existing_atoms=existing_atoms,first_pos=first_pos)
    else:
        raise Exception("ev=False is not properly implemented yet for polymer configuration generation.")
        conf = gen_polymer(num_monomer,disc_len,elems.boundary,elems.simulation_box,lb=lb,radius=0,existing_atoms=None)
    
    if centercom:
        if existing_atoms is not None and len(existing_atoms) > 0:
            print('Warning: Cannot shift the center of mass to the middle of the simulation box because the box already contains atoms.')
        elif elems.boundary == 'reflecting':
            print('Warning: Shifting of center of mass deactivated for reflecting boundary condition.')
        else:
            conf = shift2com(conf,elems.simulation_box)

    # initialize the atoms
    molecule_atoms = list()
    for i,pos in enumerate(conf):
        new_atom = elems.add_atom(sequence[i],pos)
        molecule_atoms.append(new_atom)

    #############################################
    # Bonds
    #############################################

    if bondtypename is not None and elems.types.bondtype_contained(bondtypename):
        bondtype = elems.types.bondtypes[bondtypename]
    else:
        bondstyle = bondstyle.lower() 
        # set bond coefficients
        if bondstyle == 'fene':
            coeffs =    [   ADD_POLYMER_DEFAULT_FENE_K,
                            ADD_POLYMER_DEFAULT_FENE_R0frac*disc_len,
                            ADD_POLYMER_DEFAULT_FENE_EPSILON,
                            disc_len
                        ]
        elif bondstyle == 'harmonic':
            coeffs = [  ADD_POLYMER_DEFAULT_HARMONIC_K,
                        disc_len,
                    ]
        else:
            raise ValueError(f"Unknown bondstyle '{bondstyle}'")
        # set bond type name
        if bondtypename is None:
            bondtypename = ADD_POLYMER_BASEBONDNAME
            index = 0
            while elems.types.bondtype_contained(bondtypename):
                index += 1
                bondtypename = ADD_POLYMER_BASEBONDNAME + f'#{index}'
        # if bondtype exist return that one
        if elems.types.bondtype_contained(bondtypename):
            bondtype = elems.types.bondtypes[bondtypename]
        else:
            # add bond type
            bondtype = elems.types.add_bondtype(bondtypename,bondstyle,coeffs)

    # init bonds
    molecule_bonds = list()
    for i in range(len(molecule_atoms)-1):
        atom1 = molecule_atoms[i]
        atom2 = molecule_atoms[i+1]

        atomids = [atom1.id,atom2.id]
        new_bond = elems.add_bond(bondtype.name,atomids)
        molecule_bonds.append(new_bond)

    #############################################
    # Angles
    #############################################
    molecule_angles = list()

    if bend_modulus > 0:

        angletypename = ADD_POLYMER_BASEANGLENAME
        index = 0
        while elems.types.angletype_contained(angletypename):
            index += 1
            angletypename = ADD_POLYMER_BASEANGLENAME + f'#{index}'

        coeffs = [lb/disc_len]
        angletype = elems.types.add_angletype(angletypename,ADD_POLYMER_DEFAULT_ANGLESTYLE,coeffs)
        
        # init angles
        for i in range(len(molecule_atoms)-2):
            atom1 = molecule_atoms[i]
            atom2 = molecule_atoms[i+1]
            atom3 = molecule_atoms[i+2]

            atomids     = [atom1.id,atom2.id,atom3.id]
            new_angle   = elems.add_angle(angletype.name,atomids)
            molecule_angles.append(new_angle)

    #############################################
    # Molecule
    #############################################

    if name == 'none':
        molbasename = ADD_POLYMER_BASENAME
    else:
        molbasename = str(name)
    molname = str(molbasename)
    index = 0
    while molname in elems.molecules_dict.keys():
        index += 1
        molname = molbasename + f'#{index}'

    elems.add_molecule(molname,molecule_atoms,bonds=molecule_bonds,angles=molecule_angles)


########################################################################
########################################################################
########################################################################

def shift2com(conf: np.ndarray, box: np.ndarray) -> np.ndarray:
    uconf = unwrap_polymer(conf,box)
    com = np.mean(uconf,axis=0)
    boxmiddle = 0.5*(box[:,0]+box[:,1])
    sconf = uconf + (boxmiddle-com)
    return place_in_box(box,sconf)


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

    num_monomer = 200
    disc_len    = 3.4
    sequence    = 'A'
    
    add_polymer(    elems, 
                    num_monomer, 
                    disc_len, 
                    sequence)

    for i in range(50):
        elems.add_atom("protein")

    for atom in elems.atoms:
        print(atom.id)

    filename = "test/xyz_test"
    elems.print2xyz(filename)

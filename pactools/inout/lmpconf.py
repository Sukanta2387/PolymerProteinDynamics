#!/bin/env python3

import sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy: pip install numpy")
    sys.exit("terminated")


########################################################################
########################################################################
########################################################################

# def read_conf(filename: str):

#     sim = dict()
#     with open(filename, 'r') as fob:
#         lines =  fob.readlines()
    
#     for line in lines:










def write_conf(filename: str, elements, title='LAMMPS CONF FILE'):

    atoms       = _get_attr(elements,'atoms')
    bonds       = _get_attr(elements,'bonds')
    angles      = _get_attr(elements,'angles')
    dihedrals   = _get_attr(elements,'dihedrals')
    impropers   = _get_attr(elements,'impropers')

    box = _get_attr(elements,'simulation_box')

    with open(filename, "w") as fob:

        # write header
        fob.write(f"{title}\n\n")

        # write number of atoms and interactions
        fob.write(f"   {len(atoms)} atoms\n" )
        if len(bonds) > 0:
            fob.write(f"   {len(bonds)} bonds\n" )
        if len(angles) > 0:
            fob.write(f"   {len(angles)} angles\n" )
        if len(dihedrals) > 0:
            fob.write(f"   {len(dihedrals)} dihedrals\n" )
        if len(impropers) > 0:
            fob.write(f"   {len(impropers)} impropers\n" )

        fob.write("\n")

        # # number of types
        # fob.write(f"   {_num_types(atoms)} atom types\n")
        # if len(bonds) > 0:
        #     fob.write(f"   {_num_types(bonds)} bond types\n")
        # if len(angles) > 0:
        #     fob.write(f"   {_num_types(angles)} angle types\n")
        # if len(dihedrals) > 0:
        #     fob.write(f"   {_num_types(dihedrals)} dihedral types\n")
        # if len(impropers) > 0:
        #     fob.write(f"   {_num_types(impropers)} improper types\n")

        # number of types 
        fob.write(f"   {elements.types.num_atomtypes} atom types\n")
        if len(bonds) > 0:
            fob.write(f"   {_num_types(bonds)} bond types\n")
        if len(angles) > 0:
            fob.write(f"   {_num_types(angles)} angle types\n")
        if len(dihedrals) > 0:
            fob.write(f"   {_num_types(dihedrals)} dihedral types\n")
        if len(impropers) > 0:
            fob.write(f"   {_num_types(impropers)} improper types\n")

        # # define masses
        # atomtypes = _get_atom_types(atoms)
        # fob.write("\n")
        # fob.write("Masses\n\n")
        # for atomtype in atomtypes:
        #     fob.write(f"{atomtype.id} {atomtype.mass}\n")

        # define box
        fob.write("\n")
        fob.write("# Define simulation box\n")
        fob.write(f"   {box[0,0]} {box[0,1]} xlo xhi\n")
        fob.write(f"   {box[1,0]} {box[1,1]} ylo yhi\n")
        fob.write(f"   {box[2,0]} {box[2,1]} zlo zhi\n")

        # atoms
        fob.write("\n")
        fob.write("Atoms\n\n")
        for atom in atoms:
            index   = atom.id
            type    = atom.type.id
            mol     = atom.molecule.id
            charge  = _get_attr(atom,'charge')
            # pos     = _get_attr(atom,'position')
            pos     = atom.position

            line = f"{index} {mol} {type}"
            if charge is not None:
                line += f" {charge}"
            for dim in pos:
                line += f" {dim}"
            line += "\n"
            fob.write(line)

        # velocities
        if elements.velocity_set():
            fob.write("\n")
            fob.write("Velocities\n\n")
            for atom in atoms:
                vel = atom.velocity
                line = f"{atom.id}"
                for dim in vel:
                    line += f" {dim}"
                line += "\n"
                fob.write(line)

        # bonds
        if len(bonds) > 0:
            fob.write("\n")
            fob.write("Bonds\n\n")
            for bond in bonds:
                index   = bond.id
                type    = bond.type.id
                atomids = bond.atomids
                fob.write(f"{index} {type} {atomids[0]} {atomids[1]}\n")

        # angles
        if len(angles) > 0:
            fob.write("\n")
            fob.write("Angles\n\n")
            for angle in angles:
                index   = angle.id
                type    = angle.type.id
                atomids = angle.atomids

                fob.write(f"{index} {type} {atomids[0]} {atomids[1]} {atomids[2]}\n")

        # dihedrals
        if len(dihedrals) > 0:
            fob.write("\n")
            fob.write("Dihedrals\n\n")
            for dihedral in dihedrals:
                index   = dihedral.id
                type    = dihedral.type.id
                atomids = dihedral.atomids
                fob.write(f"{index} {type} {atomids[0]} {atomids[1]} {atomids[2]} {atomids[3]}\n")

        # impropers
        if len(impropers) > 0:
            fob.write("\n")
            fob.write("Dihedrals\n\n")
            for improper in impropers:
                index   = improper.id
                type    = improper.type.id
                atomids = improper.atomids
                line = f"{index} {type} {atomids[0]}"
                for atomid in atomids:
                    line += f" {atomid}"
                line += "n"
                fob.write(line)

def _get_attr(elements,attr):
    vdict = vars(elements)
    if attr in vdict.keys():
        return vdict[attr]
    return None

def _get_atom_types(atoms):
    types = list()
    names = list()
    for atom in atoms:
        if atom.type.name not in names:
            types.append(atom.type)
            names.append(atom.type.name)
    return sorted(types, key=lambda d: d.id) 

def _num_types(entries):
    return len(list(set([entry.type.name for entry in entries])))


if __name__ == "__main__":
    raise
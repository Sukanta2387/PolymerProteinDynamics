#!/bin/env python3

import os
import sys
from typing import List, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

from pactools.interactions import lj_min
from pactools.runlmp.lmpobj import InteractionType, AtomType, BondType, AngleType, DihedralType, ImproperType, MoleculeType, Pair

########################################################################
########################################################################
########################################################################


class TypeSetup:

    ########################################################################
    # To add:
    #
    #   - remove types
    #   - remove unused types
    #   - modify types
    #
    ########################################################################

    # atomtypes = dict()
    # atomtypes_list: List[AtomType]

    # moltypes = dict()
    # moltypes_list: List[MoleculeType]

    # bondtypes = dict()
    # bondtypes_list: List[BondType]

    # angletypes = dict()
    # angletypes_list: List[AngleType]

    # dihedraltypes = dict()
    # dihedraltypes_list: List[DihedralType]

    # impropertypes = dict()
    # impropertypes_list: List[ImproperType]

    # pairs: List[Pair]

    # default_pair: Pair

    # # num_atomtypes = 0
    # # num_moltypes  = 0
    # # num_bondtypes = 0
    # # num_angletypes = 0
    # # num_pairs = 0

    # default_pair_epsilon: float
    # default_pair_style: str

    # bond_pair_exclusion: int

    ########################################################################
    ############### CONSTRUCTOR ############################################
    ########################################################################

    def __init__(self, default_pair_epsilon=1.0, default_pair_style="lj/cut",bond_pair_exclusion=1):
        self.default_pair_epsilon = default_pair_epsilon
        self.default_pair_style = default_pair_style
        self.bond_pair_exclusion = bond_pair_exclusion
        
        self.atomtypes = dict()
        self.moltypes = dict()
        self.bondtypes = dict()
        self.angletypes = dict()
        self.dihedraltypes = dict()
        self.impropertypes = dict()

        self.atomtypes_list     = list()
        self.moltypes_list      = list()
        self.bondtypes_list     = list()
        self.angletypes_list    = list()
        self.dihedraltypes_list = list()
        self.impropertypes_list = list()
        self.pairs              = list()

        self.main_pair = None
        # self._pair_lines = list()


    ########################################################################
    ############### SET ATOM AND INTERACTION TYPES #########################
    ########################################################################

    # def set_default_pair(self, epsilon: float, style: str) -> None:
    #     self.default_pair_epsilon = epsilon
    #     self.default_pair_style = style

    # def set_default_pair_epsilon(self, epsilon: float) -> None:
    #     self.default_pair_epsilon = epsilon

    # def set_default_pair_style(self, style: str) -> None:
    #     self.default_pair_style = style



    def set_main_pair(self, style: str, cutoff: float, coeffs: List[float]) -> None:
        self.main_pair = Pair('*', '*', style, cutoff, coeffs)

    def set_as_main_pair(self,name1: str, name2: str):
        pair = self.get_pair(name1,name2)
        if pair is None:
            raise ValueError(f'Pair between {name1} and {name2} does not exist.')
        self.set_main_pair(pair.style,pair.cutoff,pair.coeffs)
    

    ########################################################################
    ############### SETTERS ################################################
    ########################################################################

    def set_bond_pair_exclusion(self,bond_pair_exclusion: int) -> None:
        if bond_pair_exclusion not in [0,1,2,3]:
            raise ValueError(f'bond_pair_exclusion needs to be an integer ranging from 0 to 3. {self.bond_pair_exclusion} was given.')
        self.bond_pair_exclusion = bond_pair_exclusion

    ########################################################################

    def add_atomtype(self, name: str, radius: float, mass: float, use_main_pair=False) -> bool:
        """
        Add atom type

        returns False if atomtype name was already contained
        """
        if self.atomtype_contained(name):
            #print('ALREADY CONTAINED')
            return self.atomtypes[name]
        # self.num_atomtypes += 1
        new_atomtype = AtomType(name, mass, radius, id=self.num_atomtypes+1)
        self.atomtypes[name] = new_atomtype
        self.atomtypes_list.append(new_atomtype)
        if use_main_pair:
            if self.main_pair is None:
                print(f"Warning: main_pair not set, symmetric pair of atomtype '{name}' will be set as main_pair")
                pair = self.set_default_pair(name,name)
                self.set_as_main_pair(name,name)
        else:
            self.init_default_pairs_for_atomtype(name)
        return new_atomtype

    def atomtype_contained(self, name: str) -> bool:
        return name in self.atomtypes.keys()

    # ########################################################################

    # def add_moleculetype(self,name,bondtype_names: List[str], angletype_names: List[str]) -> bool:
    #     """
    #     Add molecule type

    #     returns False if atomtype name was already contained
    #     """
    #     if self.moleculetype_contained(name):
    #         return self.moltypes[name]
    #     self.num_moltypes += 1
    #     bondtypes  = [self.bondtypes[btn] for btn in bondtype_names]
    #     angletypes = [self.angletypes[atn] for atn in angletype_names]
    #     new_moltype = MoleculeType(name, bondtypes, angletypes, id=self.num_moltypes)
    #     self.moltypes[name] = new_moltype
    #     self.moltypes_list.append(new_moltype)
    #     return new_moltype

    # def moleculetype_contained(self, name: str) -> bool:
    #     return name in self.moltypes.keys()

    ########################################################################

    def add_bondtype(self, name: str, style: str, coeffs: List[float]) -> bool:
        """
        Add bond type

        returns False if bondtype name was already contained
        """
        if self.bondtype_contained(name):
            return self.bondtypes[name]
        # self.num_bondtypes += 1
        new_bondtype = BondType(name, style, coeffs, id=self.num_bondtypes+1)
        self.bondtypes[name] = new_bondtype
        self.bondtypes_list.append(new_bondtype)
        return new_bondtype

    def bondtype_contained(self, name: str) -> bool:
        return name in self.bondtypes.keys()

    ########################################################################

    def add_angletype(self, name: str, style: str, coeffs: List[float]) -> bool:
        """
        Add angle type

        returns False if angletype name was already contained
        """
        if self.angletype_contained(name):
            return self.angletypes[name]
        # self.num_angletypes += 1
        new_angletype = AngleType(name, style, coeffs, id=self.num_angletypes+1)
        self.angletypes[name] = new_angletype
        self.angletypes_list.append(new_angletype)
        return new_angletype

    def angletype_contained(self, name: str) -> bool:
        return name in self.angletypes.keys()

    ########################################################################

    def add_dihedraltype(self, name: str, style: str, coeffs: List[float]) -> bool:
        """
        Add dihedral type

        returns False if dihedraltype name was already contained
        """
        if self.dihedraltype_contained(name):
            return self.dihedraltypes[name]
        # self.num_dihedraltypes += 1
        new_dihedraltype = DihedralType(name, style, coeffs, id=self.num_dihedraltypes+1)
        self.dihedraltypes[name] = new_dihedraltype
        self.dihedraltypes_list.append(new_dihedraltype)
        return new_dihedraltype

    def dihedraltype_contained(self, name: str) -> bool:
        return name in self.dihedraltypes.keys()

    ########################################################################

    ########################################################################

    def add_impropertype(self, name: str, style: str, coeffs: List[float]) -> bool:
        """
        Add improper type

        returns False if impropertype name was already contained
        """
        if self.impropertype_contained(name):
            return self.impropertypes[name]
        # self.num_impropertypes += 1
        new_impropertype = ImproperType(name, style, coeffs, id=self.num_impropertypes+1)
        self.impropertypes[name] = new_impropertype
        self.impropertypes_list.append(new_impropertype)
        return new_impropertype

    def impropertype_contained(self, name: str) -> bool:
        return name in self.impropertypes.keys()

    ########################################################################


    def add_pair(
        self, name1: str, name2: str, style: str, cutoff: float, coeffs: List[float], overwrite=True
    ) -> Pair:
        """
        Adds a new pair for atoms of type name1 and name2.

        By default already existing pairs for atoms of specified types are overwritten. Overwrite can be
        deactivated by setting overwrite=False.
        """
        new_pair = Pair(name1, name2, style, cutoff, coeffs)
        if self.atomtype_contained(name1):
            new_pair.set_atomtype(self.atomtypes[name1])
        if self.atomtype_contained(name2):
            new_pair.set_atomtype(self.atomtypes[name2])
            
        if self.pair_contained(name1, name2):
            if not overwrite:
                return self.get_pair(name1,name2)
            self.pairs[self.pair_id(name1, name2)] = new_pair
        else:
            self.pairs.append(new_pair)
        # self._reset_pair_lines()
        return new_pair
        
    def set_pair_same_as_refpair(self,name1: str, name2: str, refname1: str, refname2: str) -> Pair:
        refpair = self.get_pair(refname1,refname2)
        if refpair is None:
            raise ValueError(f'Provided pair does not exist')
        return self.add_pair(name1,name2,refpair.style,refpair.cutoff,refpair.coeffs,overwrite=True)
    
    def set_pair_to_main_pair(self,name1: str, name2: str) -> None:
        if self.main_pair is None:
            raise Exception('main_pair was not yet defined!')
        pair = self.get_pair(name1,name2)
        self.pairs.remove(pair)

    def get_pair(self, name1: str, name2: str) -> Pair:
        pairid = self.pair_id(name1, name2)
        if pairid == -1:
            return None
        return self.pairs[pairid]

    def pair_contained(self, name1: str, name2: str) -> bool:
        """
        Checks if pair is already contained
        """
        if self.pair_id(name1, name2) >= 0:
            return True
        return False

    def pair_id(self, name1: str, name2: str) -> int:
        """
        Returns id of pair between atoms of type name1 and name2. If not contained
        returns -1
        """
        for i, pair in enumerate(self.pairs):
            if (pair.atom_name1 == name1 and pair.atom_name2 == name2) or (
                pair.atom_name1 == name2 and pair.atom_name2 == name1
            ):
                return i
        return -1

    def set_default_pair(self, name1: str, name2: str) -> bool:
        # print(f"Default pair: {name1} - {name2}")
        if self.pair_contained(name1, name2):
            #print("already contained")
            return False
        atype1 = self.atomtypes[name1]
        atype2 = self.atomtypes[name2]
        dist = atype1.radius + atype2.radius
        cutoff = lj_min(dist)
        coeffs = [self.default_pair_epsilon, dist]
        return self.add_pair(
            name1, name2, self.default_pair_style, cutoff, coeffs, overwrite=False
        )

    def init_default_pairs(self):
        typenames = self.atomtypes.keys()

        for i, name1 in enumerate(typenames):
            for name2 in typenames[i:]:
                self.set_default_pair(name1, name2)

    def init_default_pairs_for_atomtype(self, typename: str):
        if typename not in self.atomtypes.keys():
            return False
        for name2 in self.atomtypes.keys():
            self.set_default_pair(typename, name2)

    ########################################################################
    ############### GENERATE INPUT LINES ###################################
    ########################################################################

    def gen_input_lines(self, run_soft:bool):
        inlines = list()

        if len(self.bondtypes) > 0:
            inlines.append('### Bonds ##############################\n')
            inlines.append('\n')
            inlines.append(self._get_bond_style_line())
            for bondtype in self.bondtypes_list:
                line = f'bond_coeff {bondtype.id} {bondtype.style}'
                for coeff in bondtype.coeffs:
                    line += f' {coeff}'
                inlines.append( line + '\n' )
            inlines.append('\n')

        if len(self.angletypes) > 0:
            inlines.append('### Angles #############################\n')
            inlines.append('\n')
            inlines.append(self._get_angle_style_line())
            for angletype in self.angletypes_list:
                line = f'angle_coeff {angletype.id} {angletype.style}'
                for coeff in angletype.coeffs:
                    line += f' {coeff}'
                inlines.append( line + '\n' )
            inlines.append('\n')

        if len(self.dihedraltypes) > 0:
            inlines.append('### Dihedrals ##########################\n')
            inlines.append('\n')
            inlines.append(self._get_dihedral_style_line())
            for dihedraltype in self.dihedraltypes_list:
                line = f'dihedral_coeff {dihedraltype.id} {dihedraltype.style}'
                for coeff in dihedraltype.coeffs:
                    line += f' {coeff}'
                inlines.append( line + '\n' )
            inlines.append('\n')

        if len(self.impropertypes) > 0:
            inlines.append('### Impropers ##########################\n')
            inlines.append('\n')
            inlines.append(self._get_improper_style_line())
            for impropertype in self.impropertypes_list:
                line = f'improper_coeff {impropertype.id} {impropertype.style}'
                for coeff in impropertype.coeffs:
                    line += f' {coeff}'
                inlines.append( line + '\n' )
            inlines.append('\n')

        #if self.run

        if run_soft:
            #self.pairs=list()
            inlines.append('### Soft ##############################\n')
            inlines.append('\n')
            inlines.append(self._set_soft_interaction())
            inlines.append('\n')

        else:
            inlines.append('### Pairs ##############################\n')
            inlines.append('\n')
            inlines.append(self._get_pair_style_line())
            inlines.append('pair_modify shift yes\n')
            # for pair in self.pairs:
            #     inlines.append(self._get_pair_line(pair))
            inlines += self._get_pair_lines()
            inlines.append('\n')

        if len(self.bondtypes_list) > 0:
            if self.bond_pair_exclusion == 0:
                inlines.append('special_bonds lj 1.0 1.0 1.0\n')
            if self.bond_pair_exclusion == 1:
                inlines.append('special_bonds lj 0.0 1.0 1.0\n')
            if self.bond_pair_exclusion == 2:
                inlines.append('special_bonds lj 0.0 0.0 1.0\n')
            if self.bond_pair_exclusion == 3:
                inlines.append('special_bonds lj 0.0 0.0 0.0\n')
            inlines.append('\n')

        return inlines
    
    def _set_soft_interaction(self, cutoff=lj_min(1.0) , coeff=0.0)-> str:
        lines=''
    
        lines+='pair_style soft'
        lines+=f' {cutoff}\n'
        lines+=f'pair_coeff * * {coeff}\n'
        lines+=f'variable prefactor equal ramp(0,60)\n'
        lines+=f'fix S all adapt 1 pair soft a * * v_prefactor'

        return lines


    def _get_pair_style_line(self) -> str:
        line = 'pair_style hybrid'
        styles = self._get_styles(self.pairs)
        for style in styles:
            max_cutoff = np.max([pair.cutoff for pair in self.pairs])
            line += f' {style} {max_cutoff}'
        line += '\n'
        return line
    
    def _get_pair_line(self,pair: Pair) -> str:

        if pair.atomtype1.id < pair.atomtype2.id:
            p1id = pair.atomtype1.id
            p2id = pair.atomtype2.id
        else:
            p1id = pair.atomtype2.id
            p2id = pair.atomtype1.id
        line = f'pair_coeff {p1id} {p2id} {pair.style}'
        for coeff in pair.coeffs:
            line += f' {coeff}'
        line += f' {pair.cutoff}\n'
        return line
    
    def _get_main_pair_line(self):
        line = f'pair_coeff * * {self.main_pair.style}'
        for coeff in self.main_pair.coeffs:
            line += f' {coeff}'
        line += f' {self.main_pair.cutoff}\n'
        return line

    
    # def _reset_pair_lines(self) -> None:
    #     if len(self._pair_lines) > 0:
    #         self._pair_lines = list()

    def _get_pair_lines(self) -> List[str]:
        # if self._pair_lines > 0:
        #     return self._pair_lines
        lines = list()
        default_contained = False
        for pair in self.pairs:
            if pair == self.main_pair:
                default_contained = True
                continue
            lines.append(self._get_pair_line(pair))
        if default_contained:
            lines = [self._get_main_pair_line()] + lines
        return lines

    def _get_bond_style_line(self) -> str:
        line = 'bond_style hybrid'
        styles = self._get_styles(self.bondtypes_list)
        for style in styles:
            line += f' {style}'
        line += '\n'
        return line

    def _get_angle_style_line(self) -> str:
        line = 'angle_style hybrid'
        styles = self._get_styles(self.angletypes_list)
        for style in styles:
            line += f' {style}'
        line += '\n'
        return line

    def _get_dihedral_style_line(self) -> str:
        line = 'dihedral_style hybrid'
        styles = self._get_styles(self.dihedraltypes_list)
        for style in styles:
            line += f' {style}'
        line += '\n'
        return line

    def _get_improper_style_line(self) -> str:
        line = 'improper_style hybrid'
        styles = self._get_styles(self.impropertypes_list)
        for style in styles:
            line += f' {style}'
        line += '\n'
        return line

    def _get_styles(self,interaction_types: List[InteractionType]) -> List[str]:
        return list(set([inter.style for inter in interaction_types]))

    ########################################################################
    ###############  ###################################
    ########################################################################

    def largest_radius(self):
        #print ([at.radius for at in self.atomtypes_list])
        return np.max([at.radius for at in self.atomtypes_list])    
    @property
    def num_atomtypes(self):
        return len(self.atomtypes_list)
    
    @property
    def num_bondtypes(self):
        return len(self.bondtypes_list)
    
    @property
    def num_angletypes(self):
        return len(self.angletypes_list)
    
    @property
    def num_dihedraltypes(self):
        return len(self.dihedraltypes_list)
    
    @property
    def num_impropertypes(self):
        return len(self.impropertypes_list)




########################################################################
############### EXCEPTIONS ##############################################
########################################################################

class TypeNotDefined(Exception):

    def __init__(self, type: str, element: str, *args):
        super().__init__(args)
        self.type    = type
        self.element = element
    def __str__(self):
        return f"The {self.type} '{self.element}' was not defined."

########################################################################
########################################################################
########################################################################

if __name__ == "__main__":

    typehandler = TypeSetup()
    typehandler.add_atomtype("protein", 0.5, 1)
    typehandler.add_atomtype("dna1", 0.5, 1)
    typehandler.add_atomtype("dna2", 0.5, 1)

    max_penetration = 0.7
    ATOMPLACEMENT_PENETRATIONSTEPS = 5

    dist_scale = np.linspace(1.0, 0.5, 6)
    print(dist_scale)

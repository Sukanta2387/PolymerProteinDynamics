#!/bin/env python3

import os
import sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

from pactools.tools import conditional_jitclass

# try:
#     from numba.experimental import jitclass
#     from numba import float64
#     use_numba = True
# except ModuleNotFoundError:
#     use_numba = False


########################################################################
########################################################################
########################################################################

# @conditional_jitclass
# class Test:
#     a : int

#     def __init__(self,a: int):
#         self.a = a

class MCTypeSwitch:
    """
        If box is set, periodic boundary conditions are directly assumed
    
    """

    types: List[str]
    interactions: List[List]
    selection_probabilities: List[List]
    box:   np.ndarray

    num_types: int
    typedict: dict
    revtypedict: dict

    interaction_matrix: List[List]
    selprob_matrix: np.ndarray

    selection_pathways: List[np.ndarray]

    def __init__(self,types: List[str], interactions: List[List], default_interaction, selection_probabilities: List[List],box=None):
        self.types = types
        self.default_interaction = default_interaction
        self.box = box

        self._init_typedict(types)
        self._translate_interactions(interactions)
        self._translate_selection_probabilities(selection_probabilities)

        self._init_interaction_matrix(interactions, default_interaction)
        self._init_selprob_matrix(selection_probabilities)
    
    def _init_typedict(self,types: List[str]) -> None:
        typedict    = dict()
        revtypedict = dict()
        for i,type in enumerate(types):
            self.typedict[type] = i
            self.revtypedict[i] = type
        self.num_types = len(types)

    def _translate_interactions(self,interactions: List[List]) -> None:
        self.interactions = list()
        for interaction in interactions:
            self.interactions.append([self.typedict[interaction[0]],self.typedict[interaction[1]],interaction[2]])

    def _translate_selection_probabilities(self,selection_probabilities: List[List]) -> None:
        self.selection_probabilities = list()
        for selprob in selection_probabilities:
            self.selection_probabilities.append([self.typedict[selprob[0]],self.typedict[selprob[1]],selprob[2]])

    def _init_interaction_matrix(self) -> None:
        self.interaction_matrix = list()
        for i in range(self.num_types):
            matrix_line = list()
            for j in range(self.num_types):
                matrix_line.append(self.default_interaction)
                for interaction in self.interactions:
                    if     (interaction[0] == i and interaction[1] == j) \
                        or (interaction[0] == j and interaction[1] == i):
                        matrix_line[-1] = interaction
                        break
            self.interaction_matrix.append(matrix_line)

    def _init_selprob_matrix(self) -> None:
        self.selprob_matrix = np.zeros((self.num_types,self.num_types))
        for selprob in self.selection_probabilities:
            if selprob[2] > 1: selprob[2] = 1
            if selprob[2] < 0: selprob[2] = 0    
            self.selprob_matrix[selprob[0],selprob[1]] = selprob[2]

    def _init_selection_pathways(self, selection_probabilities: List[List]) -> None:
        self.selection_pathways = [list() for i in range(self.num_types)]
        for selprob in selection_probabilities:
            if selprob[2] > 1: selprob[2] = 1
            if selprob[2] < 0: selprob[2] = 0
            
            # check if pair already exists
            exists = False
            for elem in self.selection_pathways[self.typedict[selprob[0]]]:













########################################################################
########################################################################
########################################################################


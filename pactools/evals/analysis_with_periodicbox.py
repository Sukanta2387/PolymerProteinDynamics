#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
#import matplotlib.pyplot as plt
#import math
#import os







def unwrap_polymer(snapshot, box_length)-> np.array:

    monomer_no=np.shape(snapshot)[0]
    dimension=np.shape(snapshot)[1]
    unwrapped_polymer_profile=np.zeros((monomer_no, dimension), dtype=float)
    #print (np.shape(unwrapped_polymer_profile))
    unwrapped_polymer_profile[0]=snapshot[0]
    unwrapped_monomers=list()
    #print ((unwrapped_polymer_profile))
    for monomer in range(1, monomer_no):
        next_monomer=snapshot[monomer]
        preceding_monomer=unwrapped_polymer_profile[monomer-1] 
        #print (monomer)
        #print (next_monomer, preceding_monomer)
        updated_cord=list()
        set_of_qnts=list()
        for d in range(dimension):

            delta=next_monomer[d]-preceding_monomer[d]
            #print (delta)
            qnt=round(abs(delta)/(box_length))
            set_of_qnts.append(qnt)
            #if (qnt>1): qnt=1
            #print (qnt, delta)
            if (delta>=0.):
                 updated_cord.append(next_monomer[d]-qnt*box_length)
            else:
                updated_cord.append(next_monomer[d]+qnt*box_length)
        if 1 in set_of_qnts: unwrapped_monomers.append(monomer)
        unwrapped_polymer_profile[monomer]=updated_cord
    #print (unwrapped_monomers)
    return unwrapped_polymer_profile

def get_newbox(box_center:list, box_length:float , bothside_extension=True):
      
      newbox=np.zeros((len(box_center),2), dtype=float)
      for d in range(len(box_center)):
            if bothside_extension:
                  newbox[d][0]=box_center[d]-(box_length/2)
                  newbox[d][1]=box_center[d]+(box_length/2)

            else:
                  newbox[d][0]=box_center[d]
                  newbox[d][1]=box_center[d]+(box_length)

      return newbox

def get_HP_in_newbox(snapshot, box)->np.array:

    no_of_HP=np.shape(snapshot)[0]
    dimension=np.shape(snapshot)[1]
    HPs_inbox=np.zeros((no_of_HP, dimension), dtype=float)
    for hp in range(no_of_HP):
        for d in range(dimension):
            HPs_inbox[hp][d]=(snapshot[hp][d]-box[d][0])%(box[d][1]-box[d][0])+box[d][0]

    return HPs_inbox



if __name__ == "__main__":

    print()